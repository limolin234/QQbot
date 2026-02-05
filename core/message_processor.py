"""消息处理模块 - 封装消息监听和路由逻辑"""
import asyncio
from typing import Dict, Any
from loguru import logger
from utils.message_logger import log_received_message


def extract_message_text(message_data: Dict[str, Any]) -> str:
    """
    提取消息文本

    Args:
        message_data: 消息数据

    Returns:
        消息文本
    """
    msg = message_data.get("message", "")

    if isinstance(msg, str):
        return msg
    elif isinstance(msg, list):
        return "".join([
            seg.get("data", {}).get("text", "")
            for seg in msg if seg.get("type") == "text"
        ])
    else:
        return str(msg)


def log_message_if_needed(message_data: Dict[str, Any]):
    """
    记录消息（如果是群消息）

    Args:
        message_data: 消息数据
    """
    if message_data.get("post_type") == "message" and message_data.get("message_type") == "group":
        group_id = message_data.get("group_id")
        user_id = message_data.get("user_id")
        sender_name = message_data.get("sender", {}).get("nickname", "未知")
        msg_text = extract_message_text(message_data)
        log_received_message(group_id, user_id, sender_name, msg_text[:100])


async def message_listener(napcat_client, agent_manager):
    """
    消息监听循环

    Args:
        napcat_client: NapCat 客户端实例
        agent_manager: Agent 管理器实例
    """
    logger.info("开始监听消息...")

    try:
        async for message in napcat_client.listen():
            logger.debug(f"收到消息: {message.get('post_type')}, {message.get('message_type')}")

            # 记录收到的消息
            log_message_if_needed(message)

            # 路由消息到 Agent（异步处理，不阻塞监听）
            asyncio.create_task(agent_manager.route_message(message))

    except KeyboardInterrupt:
        logger.info("收到退出信号")
    except Exception as e:
        logger.error(f"消息监听异常: {e}", exc_info=True)
        raise
