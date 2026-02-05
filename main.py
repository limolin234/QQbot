"""QQ Bot 主程序 - 多 Agent 架构"""
import asyncio
from core.bot_initializer import initialize_bot, cleanup_bot
from core.cli_starter import start_cli
from core.message_processor import message_listener
from core.process_manager import ProcessManager


async def main():
    """主函数"""
    # 检查重复进程
    process_manager = ProcessManager()

    if not process_manager.check_and_handle_duplicates():
        return

    # 保存当前进程 PID
    process_manager.save_pid()

    try:
        # 初始化 Bot 系统
        napcat_client, agent_manager = await initialize_bot()

        if not napcat_client or not agent_manager:
            return

        # 启动 CLI 界面
        cli_manager, cli_task_or_thread = start_cli(agent_manager)

        # 监听消息
        try:
            await message_listener(napcat_client, agent_manager)
        finally:
            # 清理资源
            await cleanup_bot(napcat_client, agent_manager, cli_manager, cli_task_or_thread)

    finally:
        # 删除 PID 文件
        process_manager.remove_pid()


if __name__ == "__main__":
    asyncio.run(main())
