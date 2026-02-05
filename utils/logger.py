"""日志工具模块"""
from loguru import logger
import sys


def setup_logger(log_file="logs/bot.log", level="DEBUG"):
    """
    配置日志系统

    Args:
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        配置好的 logger 实例
    """
    # 移除默认的 handler
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # 添加文件输出
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=level,
        rotation="10 MB",  # 日志文件达到 10MB 时轮转
        retention="7 days",  # 保留 7 天的日志
        compression="zip"  # 压缩旧日志
    )

    return logger
