"""统一的 CLI 管理器 - 菜单式交互界面"""
import asyncio
import sys
import curses
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
import json
from loguru import logger


class CLIManager:
    """统一的 CLI 管理器 - 支持 interactive, panel, command 三种模式"""

    def __init__(self, agent_manager, mode: str = "interactive", refresh_rate: float = 1.0):
        """
        初始化 CLI 管理器

        Args:
            agent_manager: AgentManager 实例
            mode: 显示模式 - interactive（菜单式交互）, panel（仅面板）, command（仅命令）
            refresh_rate: 面板刷新频率（秒）
        """
        self.agent_manager = agent_manager
        self.mode = mode
        self.refresh_rate = refresh_rate
        self.console = Console()
        self.running = True

        # 菜单选项
        self.menu_items = [
            ("查看所有 Agent", self._menu_list),
            ("启用 Agent", self._menu_enable),
            ("禁用 Agent", self._menu_disable),
            ("查看 Agent 状态", self._menu_status),
            ("查看 Agent 配置", self._menu_config),
            ("查看 Agent 日志", self._menu_logs),
            ("重置 Agent 统计", self._menu_reset),
            ("退出程序", self._menu_quit),
        ]
        self.selected_index = 0
        self.scroll_offset = 0  # 输出区域的滚动偏移

        # 命令补全（仅 command 模式需要）
        if mode == "command":
            self.completer = WordCompleter([
                'help', 'list', 'enable', 'disable', 'status', 'stats',
                'config', 'logs', 'reset', 'quit', 'exit',
                'simple_chat', 'notification'
            ], ignore_case=True)
            self.session = PromptSession(completer=self.completer)

        # 最近的输出信息
        self.last_output = ""

    # ==================== 状态面板生成 ====================

    def generate_status_table(self) -> Table:
        """生成 Agent 状态表格"""
        table = Table(title="🤖 Agent 运行状态", show_header=True, header_style="bold magenta")

        table.add_column("Agent ID", style="cyan", no_wrap=True)
        table.add_column("名称", style="green")
        table.add_column("状态", justify="center")
        table.add_column("运行次数", justify="right", style="yellow")
        table.add_column("总时长", justify="right", style="blue")
        table.add_column("平均时长", justify="right", style="blue")
        table.add_column("成功", justify="right", style="green")
        table.add_column("错误", justify="right", style="red")
        table.add_column("最后运行", style="dim")

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

            avg_time = total_time / total_processed if total_processed > 0 else 0.0
            status = "🟢 运行中" if enabled else "🔴 已禁用"

            table.add_row(
                agent_id, agent_name, status, str(total_processed),
                f"{total_time:.2f}s", f"{avg_time:.3f}s",
                str(success), str(errors), last_run
            )

        return table

    # ==================== 菜单操作 ====================

    def _menu_list(self):
        """列出所有 Agent"""
        lines = ["[bold]📋 Agent 列表[/bold]\n"]
        stats_list = self.agent_manager.get_stats()
        for stats in stats_list:
            agent_id = stats.get("agent_id")
            agent_name = stats.get("agent_name")
            enabled = stats.get("enabled", True)
            total = stats.get("total_processed", 0)
            status = "🟢" if enabled else "🔴"
            lines.append(f"{status} {agent_id:15} {agent_name:10} (运行 {total} 次)")
        self.last_output = "\n".join(lines)

    def _menu_enable(self):
        """启用 Agent"""
        agent_id = self._prompt_agent_id()
        if not agent_id:
            return

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            self.last_output = f"[red]❌ Agent 不存在: {agent_id}[/red]"
            return

        if hasattr(agent, 'enabled'):
            agent.enabled = True
            self.last_output = f"[green]✅ Agent 已启用: {agent_id}[/green]"
        else:
            self.last_output = f"[yellow]⚠️  Agent 不支持启用/禁用: {agent_id}[/yellow]"

    def _menu_disable(self):
        """禁用 Agent"""
        agent_id = self._prompt_agent_id()
        if not agent_id:
            return

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            self.last_output = f"[red]❌ Agent 不存在: {agent_id}[/red]"
            return

        if hasattr(agent, 'enabled'):
            agent.enabled = False
            self.last_output = f"[yellow]🔴 Agent 已禁用: {agent_id}[/yellow]"
        else:
            self.last_output = f"[yellow]⚠️  Agent 不支持启用/禁用: {agent_id}[/yellow]"

    def _menu_status(self):
        """查看 Agent 状态"""
        agent_id = self._prompt_agent_id()
        if not agent_id:
            return

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            self.last_output = f"[red]❌ Agent 不存在: {agent_id}[/red]"
            return

        stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}
        lines = [f"[bold]📊 {stats.get('agent_name', agent_id)} 运行状态[/bold]\n"]
        lines.append(f"Agent ID: {agent_id}")
        lines.append(f"状态: {'🟢 启用' if getattr(agent, 'enabled', True) else '🔴 禁用'}")
        lines.append(f"总处理次数: {stats.get('total_processed', 0)}")
        lines.append(f"成功: [green]{stats.get('success', 0)}[/green]  错误: [red]{stats.get('errors', 0)}[/red]")
        lines.append(f"总时长: {stats.get('total_time', 0.0):.2f}s")
        lines.append(f"最后运行: {stats.get('last_run', '从未运行')}")

        self.last_output = "\n".join(lines)

    def _menu_config(self):
        """查看 Agent 配置"""
        agent_id = self._prompt_agent_id()
        if not agent_id:
            return

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            self.last_output = f"[red]❌ Agent 不存在: {agent_id}[/red]"
            return

        config = getattr(agent, 'config', {})
        config_json = json.dumps(config, indent=2, ensure_ascii=False)
        self.last_output = f"[bold]⚙️  {agent.agent_name} 配置信息[/bold]\n\n{config_json}"

    def _menu_logs(self):
        """查看 Agent 日志"""
        agent_id = self._prompt_agent_id()
        if not agent_id:
            return

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            self.last_output = f"[red]❌ Agent 不存在: {agent_id}[/red]"
            return

        agent_name = agent.agent_name

        try:
            with open("logs/message_handler.log", "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            agent_logs = [line.strip() for line in all_lines if agent_name in line]
            recent_logs = agent_logs[-10:]

            if not recent_logs:
                self.last_output = f"[yellow]⚠️  没有找到 {agent_name} 的日志[/yellow]"
                return

            result = [f"[bold]📝 {agent_name} 最近 {len(recent_logs)} 条日志[/bold]\n"]
            result.extend(recent_logs)
            self.last_output = "\n".join(result)

        except FileNotFoundError:
            self.last_output = "[red]❌ 日志文件不存在[/red]"
        except Exception as e:
            self.last_output = f"[red]❌ 读取日志失败: {e}[/red]"

    def _menu_reset(self):
        """重置 Agent 统计"""
        agent_id = self._prompt_agent_id()
        if not agent_id:
            return

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            self.last_output = f"[red]❌ Agent 不存在: {agent_id}[/red]"
            return

        if hasattr(agent, 'stats'):
            agent.stats = {
                "total_processed": 0,
                "total_time": 0.0,
                "last_run": None,
                "errors": 0,
                "success": 0
            }
            self.last_output = f"[green]✅ Agent 统计信息已重置: {agent_id}[/green]"
        else:
            self.last_output = f"[yellow]⚠️  Agent 不支持重置统计: {agent_id}[/yellow]"

    def _menu_quit(self):
        """退出程序"""
        self.last_output = "[yellow]👋 正在退出...[/yellow]"
        self.running = False

    def _prompt_agent_id(self) -> Optional[str]:
        """提示用户输入 Agent ID"""
        # 显示可用的 Agent 列表
        stats_list = self.agent_manager.get_stats()
        agent_ids = [stats.get("agent_id") for stats in stats_list]

        self.console.print("\n[bold cyan]可用的 Agent:[/bold cyan]")
        for i, agent_id in enumerate(agent_ids, 1):
            self.console.print(f"  {i}. {agent_id}")

        self.console.print("\n[dim]输入 Agent ID 或序号（按 Esc 取消）:[/dim]")

        # 读取用户输入
        try:
            user_input = input("> ").strip()
            if not user_input:
                return None

            # 如果是数字，转换为 agent_id
            if user_input.isdigit():
                index = int(user_input) - 1
                if 0 <= index < len(agent_ids):
                    return agent_ids[index]
                else:
                    self.last_output = "[red]❌ 无效的序号[/red]"
                    return None

            # 否则直接作为 agent_id
            return user_input

        except (EOFError, KeyboardInterrupt):
            return None

    # ==================== 运行模式 ====================

    async def run(self):
        """根据模式运行 CLI"""
        if self.mode == "interactive":
            await self._run_interactive()
        elif self.mode == "panel":
            await self._run_panel()
        else:  # command
            await self._run_command()

    async def _run_interactive(self):
        """交互式模式：简单菜单界面"""
        logger.info("菜单式 CLI 已启动")

        # 使用 curses 来实现菜单
        try:
            curses.wrapper(self._curses_main)
        except KeyboardInterrupt:
            logger.info("CLI 收到中断信号")
            self.running = False
        except Exception as e:
            logger.error(f"CLI 异常: {e}", exc_info=True)
            self.running = False

    def _curses_main(self, stdscr):
        """curses 主循环"""
        # 设置 curses
        curses.curs_set(0)  # 隐藏光标
        stdscr.nodelay(1)   # 非阻塞输入
        stdscr.timeout(100) # 100ms 超时

        # 初始化颜色
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)  # 选中项

        while self.running:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # 计算各区域高度
            header_height = 1
            status_height = min(len(self.agent_manager.get_stats()) + 2, 6)  # 最多显示6行
            menu_height = len(self.menu_items) + 2
            help_height = 1
            output_height = max(height - header_height - status_height - menu_height - help_height - 2, 5)

            current_row = 0

            # ========== 绘制标题 ==========
            title = "🎮 QQ Bot 菜单式控制面板"
            try:
                stdscr.addstr(current_row, max(0, (width - len(title)) // 2), title[:width-1],
                             curses.color_pair(1) | curses.A_BOLD)
            except curses.error:
                pass
            current_row += 1

            # ========== 绘制 Agent 状态 ==========
            current_row += 1
            try:
                stdscr.addstr(current_row, 2, "🤖 Agent 运行状态", curses.color_pair(2) | curses.A_BOLD)
            except curses.error:
                pass
            current_row += 1

            stats_list = self.agent_manager.get_stats()
            for i, stats in enumerate(stats_list):
                if i >= status_height - 2:  # 限制显示行数
                    break
                agent_id = stats.get("agent_id", "unknown")
                agent_name = stats.get("agent_name", "Unknown")
                enabled = stats.get("enabled", True)
                total = stats.get("total_processed", 0)
                success = stats.get("success", 0)
                errors = stats.get("errors", 0)
                status = "🟢" if enabled else "🔴"

                line = f"{status} {agent_id:15} {agent_name:15} 运行:{total:3} 成功:{success:3} 错误:{errors:2}"
                try:
                    stdscr.addstr(current_row, 4, line[:width-5])
                except curses.error:
                    pass
                current_row += 1

            # ========== 绘制菜单 ==========
            current_row += 1
            try:
                stdscr.addstr(current_row, 2, "📋 操作菜单", curses.color_pair(3) | curses.A_BOLD)
            except curses.error:
                pass
            current_row += 1

            for i, (label, _) in enumerate(self.menu_items):
                try:
                    if i == self.selected_index:
                        stdscr.addstr(current_row, 4, f"▶ {label}"[:width-5],
                                     curses.color_pair(5) | curses.A_BOLD)
                    else:
                        stdscr.addstr(current_row, 4, f"  {label}"[:width-5])
                except curses.error:
                    pass
                current_row += 1

            # ========== 绘制提示 ==========
            current_row += 1
            help_text = "💡 ↑↓选择 ↩确认 q退出 | 输出区: PgUp/PgDn滚动"
            try:
                stdscr.addstr(current_row, 2, help_text[:width-3], curses.A_DIM)
            except curses.error:
                pass
            current_row += 1

            # ========== 绘制输出区域（支持滚动）==========
            if self.last_output:
                try:
                    stdscr.addstr(current_row, 2, "📤 输出:", curses.color_pair(3))
                except curses.error:
                    pass
                current_row += 1

                # 清理输出文本（移除 rich 标记）
                output_text = self.last_output
                for tag in ["[bold]", "[/bold]", "[green]", "[/green]", "[red]", "[/red]",
                           "[yellow]", "[/yellow]", "[cyan]", "[/cyan]", "[dim]", "[/dim]"]:
                    output_text = output_text.replace(tag, "")

                output_lines = output_text.split("\n")
                total_output_lines = len(output_lines)

                # 计算可显示的行数
                available_lines = height - current_row - 1

                # 调整滚动偏移
                max_scroll = max(0, total_output_lines - available_lines)
                self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

                # 显示输出（带滚动）
                for i in range(available_lines):
                    line_index = i + self.scroll_offset
                    if line_index < total_output_lines:
                        line = output_lines[line_index]
                        try:
                            stdscr.addstr(current_row + i, 4, line[:width-5])
                        except curses.error:
                            pass

                # 显示滚动指示器
                if total_output_lines > available_lines:
                    scroll_info = f"[{self.scroll_offset + 1}-{min(self.scroll_offset + available_lines, total_output_lines)}/{total_output_lines}]"
                    try:
                        stdscr.addstr(height - 1, width - len(scroll_info) - 2, scroll_info, curses.A_DIM)
                    except curses.error:
                        pass

            stdscr.refresh()

            # ========== 处理键盘输入 ==========
            key = stdscr.getch()
            if key == curses.KEY_UP:
                self.selected_index = (self.selected_index - 1) % len(self.menu_items)
            elif key == curses.KEY_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.menu_items)
            elif key == curses.KEY_PPAGE:  # Page Up - 向上滚动输出
                self.scroll_offset = max(0, self.scroll_offset - 5)
            elif key == curses.KEY_NPAGE:  # Page Down - 向下滚动输出
                if self.last_output:
                    output_lines = self.last_output.split("\n")
                    available_lines = height - current_row - 1
                    max_scroll = max(0, len(output_lines) - available_lines)
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 5)
            elif key == ord('\n') or key == ord('\r'):
                # 执行选中的菜单项
                _, action = self.menu_items[self.selected_index]
                curses.endwin()  # 暂时退出 curses
                self.scroll_offset = 0  # 重置滚动
                action()
                stdscr = curses.initscr()  # 重新初始化
                curses.curs_set(0)
                stdscr.nodelay(1)
                stdscr.timeout(100)
                curses.start_color()
                curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
                curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
                curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
                curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
                curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)
            elif key == ord('q') or key == ord('Q'):
                self.running = False

    async def _run_panel(self):
        """仅面板模式：只显示状态"""
        from rich.live import Live
        logger.info("CLI 控制面板已启动")

        try:
            with Live(self.generate_status_table(), refresh_per_second=1/self.refresh_rate, console=self.console) as live:
                while self.running:
                    await asyncio.sleep(self.refresh_rate)
                    live.update(self.generate_status_table())
        except KeyboardInterrupt:
            logger.info("CLI 控制面板收到中断信号")
            self.running = False
        except Exception as e:
            logger.error(f"CLI 控制面板异常: {e}", exc_info=True)
            self.running = False

    async def _run_command(self):
        """仅命令模式：只有命令行"""
        logger.info("CLI 命令行模式已启动")
        self.console.print("[bold green]🎮 QQ Bot CLI 已启动[/bold green]")
        self.console.print("[dim]输入 'help' 查看可用命令，输入 'quit' 退出[/dim]\n")

        while self.running:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.prompt("\n[Bot] > ")
                )
                await self.process_command(command)

            except EOFError:
                break
            except KeyboardInterrupt:
                self.console.print("\n[yellow]使用 'quit' 命令退出[/yellow]")
            except Exception as e:
                logger.error(f"CLI 异常: {e}", exc_info=True)

    async def process_command(self, command_line: str):
        """处理命令并更新输出"""
        if not command_line.strip():
            return

        parts = command_line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        output_lines = []

        try:
            if cmd == "help":
                output_lines = self._cmd_help()
            elif cmd == "list":
                output_lines = self._cmd_list()
            elif cmd == "enable":
                output_lines = self._cmd_enable(args)
            elif cmd == "disable":
                output_lines = self._cmd_disable(args)
            elif cmd in ["status", "stats"]:
                output_lines = self._cmd_status(args)
            elif cmd == "config":
                output_lines = self._cmd_config(args)
            elif cmd == "logs":
                output_lines = self._cmd_logs(args)
            elif cmd == "reset":
                output_lines = self._cmd_reset(args)
            elif cmd in ["quit", "exit"]:
                output_lines = ["[yellow]👋 正在退出...[/yellow]"]
                self.running = False
            else:
                output_lines = [
                    f"[red]❌ 未知命令: {cmd}[/red]",
                    "[dim]输入 'help' 查看可用命令[/dim]"
                ]

            # 直接打印输出（command 模式）
            for line in output_lines:
                self.console.print(line)

        except Exception as e:
            error_msg = f"[red]❌ 命令执行失败: {e}[/red]"
            self.console.print(error_msg)
            logger.error(f"命令执行异常: {e}", exc_info=True)

    def _cmd_help(self) -> List[str]:
        """帮助命令"""
        return [
            "[bold cyan]📋 可用命令[/bold cyan]\n",
            "[yellow]基础命令：[/yellow]",
            "  help                    - 显示此帮助信息",
            "  list                    - 列出所有 Agent",
            "  quit / exit             - 退出程序\n",
            "[yellow]Agent 控制：[/yellow]",
            "  enable <agent_id>       - 启用指定 Agent",
            "  disable <agent_id>      - 禁用指定 Agent",
            "  status <agent_id>       - 查看 Agent 运行状态",
            "  stats <agent_id>        - 查看 Agent 统计信息",
            "  reset <agent_id>        - 重置 Agent 统计信息\n",
            "[yellow]配置和日志：[/yellow]",
            "  config <agent_id>       - 查看 Agent 配置信息",
            "  logs <agent_id> [n]     - 查看 Agent 最近 n 条日志"
        ]

    def _cmd_list(self) -> List[str]:
        """列出所有 Agent"""
        lines = ["[bold]📋 Agent 列表[/bold]\n"]
        stats_list = self.agent_manager.get_stats()
        for stats in stats_list:
            agent_id = stats.get("agent_id")
            agent_name = stats.get("agent_name")
            enabled = stats.get("enabled", True)
            total = stats.get("total_processed", 0)
            status = "🟢" if enabled else "🔴"
            lines.append(f"{status} {agent_id:15} {agent_name:10} (运行 {total} 次)")
        return lines

    def _cmd_enable(self, args: List[str]) -> List[str]:
        """启用 Agent"""
        if not args:
            return ["[red]❌ 错误：请指定 Agent ID[/red]", "[dim]用法：enable <agent_id>[/dim]"]

        agent_id = args[0]
        agent = self.agent_manager.get_agent(agent_id)

        if not agent:
            return [f"[red]❌ Agent 不存在: {agent_id}[/red]"]

        if hasattr(agent, 'enabled'):
            agent.enabled = True
            return [f"[green]✅ Agent 已启用: {agent_id}[/green]"]
        else:
            return [f"[yellow]⚠️  Agent 不支持启用/禁用: {agent_id}[/yellow]"]

    def _cmd_disable(self, args: List[str]) -> List[str]:
        """禁用 Agent"""
        if not args:
            return ["[red]❌ 错误：请指定 Agent ID[/red]", "[dim]用法：disable <agent_id>[/dim]"]

        agent_id = args[0]
        agent = self.agent_manager.get_agent(agent_id)

        if not agent:
            return [f"[red]❌ Agent 不存在: {agent_id}[/red]"]

        if hasattr(agent, 'enabled'):
            agent.enabled = False
            return [f"[yellow]🔴 Agent 已禁用: {agent_id}[/yellow]"]
        else:
            return [f"[yellow]⚠️  Agent 不支持启用/禁用: {agent_id}[/yellow]"]

    def _cmd_status(self, args: List[str]) -> List[str]:
        """查看 Agent 状态"""
        if not args:
            return ["[red]❌ 错误：请指定 Agent ID[/red]", "[dim]用法：status <agent_id>[/dim]"]

        agent_id = args[0]
        agent = self.agent_manager.get_agent(agent_id)

        if not agent:
            return [f"[red]❌ Agent 不存在: {agent_id}[/red]"]

        stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}
        lines = [f"[bold]📊 {stats.get('agent_name', agent_id)} 运行状态[/bold]\n"]
        lines.append(f"Agent ID: {agent_id}")
        lines.append(f"状态: {'🟢 启用' if getattr(agent, 'enabled', True) else '🔴 禁用'}")
        lines.append(f"总处理次数: {stats.get('total_processed', 0)}")
        lines.append(f"成功: [green]{stats.get('success', 0)}[/green]  错误: [red]{stats.get('errors', 0)}[/red]")
        lines.append(f"总时长: {stats.get('total_time', 0.0):.2f}s")
        lines.append(f"最后运行: {stats.get('last_run', '从未运行')}")

        return lines

    def _cmd_config(self, args: List[str]) -> List[str]:
        """查看 Agent 配置"""
        if not args:
            return ["[red]❌ 错误：请指定 Agent ID[/red]", "[dim]用法：config <agent_id>[/dim]"]

        agent_id = args[0]
        agent = self.agent_manager.get_agent(agent_id)

        if not agent:
            return [f"[red]❌ Agent 不存在: {agent_id}[/red]"]

        config = getattr(agent, 'config', {})
        config_json = json.dumps(config, indent=2, ensure_ascii=False)

        return [f"[bold]⚙️  {agent.agent_name} 配置信息[/bold]\n", config_json]

    def _cmd_logs(self, args: List[str]) -> List[str]:
        """查看 Agent 日志"""
        if not args:
            return ["[red]❌ 错误：请指定 Agent ID[/red]", "[dim]用法：logs <agent_id> [行数][/dim]"]

        agent_id = args[0]
        lines_count = int(args[1]) if len(args) > 1 else 10

        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            return [f"[red]❌ Agent 不存在: {agent_id}[/red]"]

        agent_name = agent.agent_name

        try:
            with open("logs/message_handler.log", "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            agent_logs = [line.strip() for line in all_lines if agent_name in line]
            recent_logs = agent_logs[-lines_count:]

            if not recent_logs:
                return [f"[yellow]⚠️  没有找到 {agent_name} 的日志[/yellow]"]

            result = [f"[bold]📝 {agent_name} 最近 {len(recent_logs)} 条日志[/bold]\n"]
            result.extend(recent_logs)
            return result

        except FileNotFoundError:
            return ["[red]❌ 日志文件不存在[/red]"]
        except Exception as e:
            return [f"[red]❌ 读取日志失败: {e}[/red]"]

    def _cmd_reset(self, args: List[str]) -> List[str]:
        """重置 Agent 统计"""
        if not args:
            return ["[red]❌ 错误：请指定 Agent ID[/red]", "[dim]用法：reset <agent_id>[/dim]"]

        agent_id = args[0]
        agent = self.agent_manager.get_agent(agent_id)

        if not agent:
            return [f"[red]❌ Agent 不存在: {agent_id}[/red]"]

        if hasattr(agent, 'stats'):
            agent.stats = {
                "total_processed": 0,
                "total_time": 0.0,
                "last_run": None,
                "errors": 0,
                "success": 0
            }
            return [f"[green]✅ Agent 统计信息已重置: {agent_id}[/green]"]
        else:
            return [f"[yellow]⚠️  Agent 不支持重置统计: {agent_id}[/yellow]"]

    def stop(self):
        """停止 CLI"""
        self.running = False
        logger.info("CLI 已停止")
