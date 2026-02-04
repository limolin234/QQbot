"""QQ Bot 主程序"""
import os
import asyncio
from dotenv import load_dotenv
from loguru import logger

from config.napcat_config import NAPCAT_WS_URL
from core.napcat_client import NapCatClient
from core.message_filter import MessageFilter
from core.message_handler import MessageHandler
from agents.simple_chat_agent import SimpleChatAgent
from utils.logger import setup_logger
from utils.message_logger import setup_message_logger


async def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()

    # 设置日志
    setup_logger()

    # 设置消息处理日志（必须在 setup_logger 之后）
    setup_message_logger()

    logger.info("=" * 50)
    logger.info("QQ Bot 启动中...")
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

    # 初始化 Agent
    logger.info("初始化 LangGraph Agent...")
    agent = SimpleChatAgent(api_key, base_url)

    # 初始化消息过滤器
    message_filter = MessageFilter(napcat_client.bot_qq)

    # 初始化消息处理器
    message_handler = MessageHandler(agent, napcat_client, message_filter)

    logger.success("所有组件初始化完成")
    logger.info("开始监听消息...")

    # 监听消息循环
    try:
        async for message in napcat_client.listen():
            # 异步处理消息（不阻塞监听）
            asyncio.create_task(message_handler.handle_message(message))

    except KeyboardInterrupt:
        logger.info("收到退出信号")
    except Exception as e:
        logger.error(f"运行异常: {e}", exc_info=True)
    finally:
        # 清理资源
        await napcat_client.close()
        logger.info("Bot 已停止")


if __name__ == "__main__":
    asyncio.run(main())
