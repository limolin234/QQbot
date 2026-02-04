"""CLI 控制面板 - 实时显示 Agent 运行状态"""
import asyncio
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from loguru import logger


class CLIPanel:
    """CLI 控制面板 - 使用 rich 库实时显示 Agent 状态"""

    def __init__(self, agent_manager, refresh_rate: float = 1.0):
        """
        初始化 CLI 面板

        Args:
            agent_manager: AgentManager 实例
            refresh_rate: 刷新频率（秒）
        """
        self.agent_manager = agent_manager
        self.refresh_rate = refresh_rate
        self.console = Console()
        self.running = False

    def generate_status_table(self) -> Table:
        """生成 Agent 状态表格"""
        table = Table(title="🤖 Agent 运行状态", show_header=True, header_style="bold magenta")

        # 添加列
        table.add_column("Agent ID", style="cyan", no_wrap=True)
        table.add_column("名称", style="green")
        table.add_column("状态", justify="center")
        table.add_column("运行次数", justify="right", style="yellow")
        table.add_column("总时长", justify="right", style="blue")
        table.add_column("平均时长", justify="right", style="blue")
        table.add_column("成功", justify="right", style="green")
        table.add_column("错误", justify="right", style="red")
        table.add_column("最后运行", style="dim")

        # 获取所有 Agent 的统计信息
        stats_list = self.agent_manager.get_stats()

        for stats in stats_list:
            agent_id = stats.get("agent_id", "unknown")
            agent_name = stats.get("agent_name", "Unknown")
            enabled = stats.get("enabled", True)
            total_processed = stats.get("total_processed", 0)
            total_time = stats.get("total_time", 0.0)
            success = stats.get("success", 0)
            errors = stats.get("errors", 0)
            last_run = stats.get("last_run", "从未运行")

            # 计算平均时长
            avg_time = total_time / total_processed if total_processed > 0 else 0.0

            # 状态显示
            status = "🟢 运行中" if enabled else "🔴 已禁用"

            # 添加行
            table.add_row(
                agent_id,
                agent_name,
                status,
                str(total_processed),
                f"{total_time:.2f}s",
                f"{avg_time:.3f}s",
                str(success),
                str(errors),
                last_run
            )

        return table

    def generate_help_panel(self) -> Panel:
        """生成帮助面板"""
        help_text = Text()
        help_text.append("📋 控制命令\n\n", style="bold cyan")
        help_text.append("• Ctrl+C - 退出程序\n", style="dim")
        help_text.append("• 面板每 ", style="dim")
        help_text.append(f"{self.refresh_rate}", style="yellow")
        help_text.append(" 秒自动刷新\n", style="dim")

        return Panel(help_text, title="帮助", border_style="blue")

    def generate_layout(self) -> Layout:
        """生成布局"""
        layout = Layout()

        # 分割布局
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5)
        )

        # 设置各部分内容
        layout["header"].update(Panel("🎮 QQ Bot 多 Agent 控制面板", style="bold white on blue"))
        layout["body"].update(self.generate_status_table())
        layout["footer"].update(self.generate_help_panel())

        return layout

    async def run(self):
        """运行控制面板（实时刷新模式）"""
        self.running = True
        logger.info("CLI 控制面板已启动")

        try:
            with Live(self.generate_layout(), refresh_per_second=1/self.refresh_rate, console=self.console) as live:
                while self.running:
                    await asyncio.sleep(self.refresh_rate)
                    live.update(self.generate_layout())
        except KeyboardInterrupt:
            logger.info("CLI 控制面板收到中断信号")
            self.running = False
        except Exception as e:
            logger.error(f"CLI 控制面板异常: {e}", exc_info=True)
            self.running = False

    def stop(self):
        """停止控制面板"""
        self.running = False
        logger.info("CLI 控制面板已停止")

    def print_stats(self):
        """打印统计信息（一次性显示）"""
        self.console.print(self.generate_status_table())

    def print_agent_detail(self, agent_id: str):
        """
        打印单个 Agent 的详细信息

        Args:
            agent_id: Agent ID
        """
        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            self.console.print(f"[red]Agent 不存在: {agent_id}[/red]")
            return

        stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}

        # 创建详细信息表格
        table = Table(title=f"Agent 详细信息: {agent_id}", show_header=False)
        table.add_column("属性", style="cyan")
        table.add_column("值", style="yellow")

        table.add_row("Agent ID", stats.get("agent_id", agent_id))
        table.add_row("名称", stats.get("agent_name", "Unknown"))
        table.add_row("状态", "启用" if getattr(agent, 'enabled', True) else "禁用")
        table.add_row("总处理次数", str(stats.get("total_processed", 0)))
        table.add_row("总时长", f"{stats.get('total_time', 0.0):.2f}s")
        table.add_row("平均时长", f"{stats.get('avg_time', 0.0):.3f}s")
        table.add_row("成功次数", str(stats.get("success", 0)))
        table.add_row("错误次数", str(stats.get("errors", 0)))
        table.add_row("最后运行", stats.get("last_run", "从未运行"))

        # 如果是 NotificationAgent，显示额外信息
        if hasattr(agent, 'stats') and "important_notifications" in agent.stats:
            table.add_row("重要通知数", str(agent.stats.get("important_notifications", 0)))
            table.add_row("已发送摘要", str(agent.stats.get("sent_summaries", 0)))

        self.console.print(table)
