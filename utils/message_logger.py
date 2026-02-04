"""消息处理日志工具 - 专门记录 Agent 处理的有效消息"""
from loguru import logger
import sys


# 创建独立的消息处理 logger
message_logger = logger.bind(context="message_handler")


def setup_message_logger(log_file="logs/message_handler.log", level="INFO"):
    """
    配置消息处理专用日志

    注意：必须在 setup_logger() 之后调用

    Args:
        log_file: 日志文件路径
        level: 日志级别
    """
    # 添加文件输出（只记录消息处理相关的日志）
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | [消息处理] | {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        filter=lambda record: record["extra"].get("context") == "message_handler"
    )
    logger.info(f"消息处理日志已配置: {log_file}")


def log_received_message(group_id: int, user_id: int, user_name: str, message: str):
    """记录收到的有效消息"""
    message_logger.info(
        f"📨 收到消息 | 群:{group_id} | 用户:{user_name}({user_id}) | 内容:{message[:100]}"
    )


def log_agent_processing(group_id: int, user_id: int, message: str):
    """记录 Agent 开始处理"""
    message_logger.info(
        f"🤖 Agent处理中 | 群:{group_id} | 用户:{user_id} | 消息:{message[:50]}"
    )


def log_agent_response(group_id: int, user_id: int, response: str):
    """记录 Agent 生成的回复"""
    message_logger.info(
        f"💬 Agent回复 | 群:{group_id} | 用户:{user_id} | 回复:{response[:100]}"
    )


def log_message_sent(group_id: int, success: bool):
    """记录消息发送结果"""
    if success:
        message_logger.success(f"✅ 消息已发送 | 群:{group_id}")
    else:
        message_logger.error(f"❌ 消息发送失败 | 群:{group_id}")


def log_security_block(group_id: int, reason: str):
    """记录安全拦截"""
    message_logger.warning(f"🚫 安全拦截 | 群:{group_id} | 原因:{reason}")


def log_filter_skip(group_id: int, reason: str):
    """记录过滤跳过"""
    message_logger.debug(f"⏭️  消息跳过 | 群:{group_id} | 原因:{reason}")
