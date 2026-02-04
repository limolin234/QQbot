"""QQ Bot 主程序 - 多 Agent 架构"""
import os
import asyncio
from dotenv import load_dotenv
from loguru import logger

from config.napcat_config import NAPCAT_WS_URL
from config.agents_config import AGENTS_CONFIG, CLI_PANEL_CONFIG
from core.napcat_client import NapCatClient
from core.agent_manager import AgentManager
from agents.simple_chat_agent import SimpleChatAgent
from agents.notification_agent import NotificationAgent
from utils.logger import setup_logger
from utils.message_logger import setup_message_logger, log_received_message
from utils.cli_panel import CLIPanel


async def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()

    # 设置日志
    setup_logger()

    # 设置消息处理日志（必须在 setup_logger 之后）
    setup_message_logger()

    logger.info("=" * 50)
    logger.info("🤖 QQ Bot 多 Agent 系统启动中...")
    logger.info("=" * 50)

    # 读取配置
    api_key = os.getenv("YUNWU_API_KEY")
    base_url = os.getenv("API_BASE_URL")

    if not api_key or not base_url:
        logger.error("请在 .env 文件中配置 YUNWU_API_KEY 和 API_BASE_URL")
        return

    # 初始化 NapCat 客户端
    napcat_client = NapCatClient(NAPCAT_WS_URL)

    # 连接到 NapCat
    if not await napcat_client.connect():
        logger.error("无法连接到 NapCat，请检查 NapCat 是否运行以及配置是否正确")
        return

    # 获取 bot QQ 号
    if not napcat_client.bot_qq:
        logger.error("无法获取 Bot QQ 号")
        return

    logger.info(f"Bot QQ: {napcat_client.bot_qq}")

    # 保存 bot QQ 到配置（用于消息过滤）
    from config import bot_config
    bot_config.BOT_QQ = napcat_client.bot_qq

    # 初始化 AgentManager
    logger.info("初始化 Agent 管理器...")
    agent_manager = AgentManager(napcat_client)

    # 加载所有 Agent
    logger.info("加载 Agent...")
    for agent_id, agent_config in AGENTS_CONFIG.items():
        if not agent_config.get("enabled", True):
            logger.info(f"跳过已禁用的 Agent: {agent_id}")
            continue

        try:
            agent_class_name = agent_config.get("class", "")
            config = agent_config.get("config", {})

            # 根据类名创建 Agent 实例
            if agent_class_name == "SimpleChatAgent":
                agent = SimpleChatAgent(
                    agent_id=agent_id,
                    config={**agent_config, **config},
                    api_key=api_key,
                    base_url=base_url,
                    napcat_client=napcat_client
                )
            elif agent_class_name == "NotificationAgent":
                agent = NotificationAgent(
                    agent_id=agent_id,
                    config={**agent_config, **config},
                    api_key=api_key,
                    base_url=base_url,
                    napcat_client=napcat_client
                )
            else:
                logger.error(f"未知的 Agent 类型: {agent_class_name}")
                continue

            # 注册 Agent
            agent_manager.register_agent(agent_id, agent)
            logger.success(f"Agent 已加载: {agent_id} ({agent.agent_name})")

        except Exception as e:
            logger.error(f"加载 Agent 失败 ({agent_id}): {e}", exc_info=True)

    # 启动所有 Agent 的 worker
    logger.info("启动 Agent workers...")
    agent_manager.start_workers()

    logger.success("所有组件初始化完成")
    logger.info("开始监听消息...")

    # 创建 CLI 面板
    cli_panel = None
    cli_task = None
    if CLI_PANEL_CONFIG.get("enabled", True):
        cli_panel = CLIPanel(
            agent_manager,
            refresh_rate=CLI_PANEL_CONFIG.get("refresh_rate", 1)
        )
        # 启动 CLI 面板（异步任务）
        cli_task = asyncio.create_task(cli_panel.run())
        logger.info("CLI 控制面板已启动")

    # 监听消息循环
    try:
        async for message in napcat_client.listen():
            # 记录收到的消息（如果是群消息）
            if message.get("post_type") == "message" and message.get("message_type") == "group":
                group_id = message.get("group_id")
                user_id = message.get("user_id")
                sender_name = message.get("sender", {}).get("nickname", "未知")

                # 提取消息文本
                msg = message.get("message", "")
                if isinstance(msg, str):
                    msg_text = msg
                elif isinstance(msg, list):
                    msg_text = "".join([
                        seg.get("data", {}).get("text", "")
                        for seg in msg if seg.get("type") == "text"
                    ])
                else:
                    msg_text = str(msg)

                log_received_message(group_id, user_id, sender_name, msg_text[:100])

            # 路由消息到 Agent（异步处理，不阻塞监听）
            asyncio.create_task(agent_manager.route_message(message))

    except KeyboardInterrupt:
        logger.info("收到退出信号")
    except Exception as e:
        logger.error(f"运行异常: {e}", exc_info=True)
    finally:
        # 清理资源
        logger.info("正在关闭...")

        # 停止 CLI 面板
        if cli_panel:
            cli_panel.stop()
        if cli_task:
            cli_task.cancel()
            try:
                await cli_task
            except asyncio.CancelledError:
                pass

        # 关闭 AgentManager
        await agent_manager.shutdown()

        # 关闭 NapCat 客户端
        await napcat_client.close()

        logger.info("Bot 已停止")


if __name__ == "__main__":
    asyncio.run(main())
