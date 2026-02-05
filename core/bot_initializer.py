"""Bot 初始化模块 - 封装启动逻辑"""
import os
from loguru import logger
from dotenv import load_dotenv

from config.napcat_config import NAPCAT_WS_URL
from config.agents_config import AGENTS_CONFIG
from core.napcat_client import NapCatClient
from core.agent_manager import AgentManager
from core.agent_factory import AgentFactory
from utils.logger import setup_logger
from utils.message_logger import setup_message_logger

# 导入所有 Agent 类（触发装饰器注册）
from agents.simple_chat_agent import SimpleChatAgent
from agents.notification_agent import NotificationAgent


async def initialize_bot():
    """
    初始化 Bot 系统

    Returns:
        tuple: (napcat_client, agent_manager) 或 (None, None) 如果初始化失败
    """
    # 加载环境变量
    load_dotenv()

    # 设置日志
    setup_logger()
    setup_message_logger()

    logger.info("=" * 50)
    logger.info("🤖 QQ Bot 多 Agent 系统启动中...")
    logger.info("=" * 50)

    # 读取 API 配置
    api_key = os.getenv("YUNWU_API_KEY")
    base_url = os.getenv("API_BASE_URL")

    if not api_key or not base_url:
        logger.error("请在 .env 文件中配置 YUNWU_API_KEY 和 API_BASE_URL")
        return None, None

    # 初始化 NapCat 客户端
    napcat_client = NapCatClient(NAPCAT_WS_URL)

    # 连接到 NapCat
    if not await napcat_client.connect():
        logger.error("无法连接到 NapCat，请检查 NapCat 是否运行以及配置是否正确")
        return None, None

    # 获取 bot QQ 号
    if not napcat_client.bot_qq:
        logger.error("无法获取 Bot QQ 号")
        await napcat_client.close()
        return None, None

    logger.info(f"Bot QQ: {napcat_client.bot_qq}")

    # 保存 bot QQ 到配置（用于消息过滤）
    from config import bot_config
    bot_config.BOT_QQ = napcat_client.bot_qq

    # 初始化 AgentManager
    logger.info("初始化 Agent 管理器...")
    agent_manager = AgentManager(napcat_client)

    # 加载所有 Agent
    logger.info("加载 Agent...")
    loaded_count = AgentFactory.load_all_agents(
        agents_config=AGENTS_CONFIG,
        api_key=api_key,
        base_url=base_url,
        napcat_client=napcat_client,
        agent_manager=agent_manager
    )

    if loaded_count == 0:
        logger.warning("没有加载任何 Agent")
    else:
        logger.success(f"成功加载 {loaded_count} 个 Agent")

    # 启动所有 Agent 的 worker
    logger.info("启动 Agent workers...")
    agent_manager.start_workers()

    logger.success("所有组件初始化完成")

    return napcat_client, agent_manager


async def cleanup_bot(napcat_client, agent_manager, cli_manager=None, cli_task_or_thread=None):
    """
    清理 Bot 资源

    Args:
        napcat_client: NapCat 客户端实例
        agent_manager: Agent 管理器实例
        cli_manager: CLI 管理器实例（可选）
        cli_task_or_thread: CLI 任务或线程（可选）
    """
    logger.info("正在关闭...")

    # 停止 CLI
    if cli_manager:
        cli_manager.stop()

    if cli_task_or_thread:
        # 如果是线程，等待线程结束
        if isinstance(cli_task_or_thread, __import__('threading').Thread):
            logger.debug("等待 CLI 线程结束...")
            cli_task_or_thread.join(timeout=2)
        # 如果是任务，取消任务
        else:
            cli_task_or_thread.cancel()
            try:
                await cli_task_or_thread
            except Exception:
                pass

    # 关闭 AgentManager
    if agent_manager:
        await agent_manager.shutdown()

    # 关闭 NapCat 客户端
    if napcat_client:
        await napcat_client.close()

    logger.info("Bot 已停止")
