"""CLI 启动模块 - 封装 CLI 初始化逻辑"""
import asyncio
import threading
from loguru import logger
from config.agents_config import CLI_PANEL_CONFIG
from utils.cli_manager import CLIManager


def start_cli(agent_manager):
    """
    启动 CLI 界面（在单独的线程中运行）

    Args:
        agent_manager: Agent 管理器实例

    Returns:
        tuple: (cli_manager, cli_thread) 或 (None, None) 如果 CLI 未启用
    """
    if not CLI_PANEL_CONFIG.get("enabled", True):
        logger.info("CLI 已禁用")
        return None, None

    cli_mode = CLI_PANEL_CONFIG.get("mode", "interactive")
    refresh_rate = CLI_PANEL_CONFIG.get("refresh_rate", 1)

    # 如果是 interactive 模式，使用线程运行 curses
    if cli_mode == "interactive":
        cli_manager = CLIManager(
            agent_manager=agent_manager,
            mode=cli_mode,
            refresh_rate=refresh_rate
        )

        # 在单独的线程中运行 CLI（curses 是同步的）
        def run_cli_thread():
            import curses
            try:
                curses.wrapper(cli_manager._curses_main)
            except KeyboardInterrupt:
                logger.info("CLI 线程收到中断信号")
            except Exception as e:
                logger.error(f"CLI 线程异常: {e}", exc_info=True)

        cli_thread = threading.Thread(target=run_cli_thread, daemon=True)
        cli_thread.start()
        logger.info(f"CLI 已启动（模式: {cli_mode}，运行在独立线程）")

        return cli_manager, cli_thread

    else:
        # panel 和 command 模式使用异步任务
        cli_manager = CLIManager(
            agent_manager=agent_manager,
            mode=cli_mode,
            refresh_rate=refresh_rate
        )

        cli_task = asyncio.create_task(cli_manager.run())
        logger.info(f"CLI 已启动（模式: {cli_mode}）")

        return cli_manager, cli_task
