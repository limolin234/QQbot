"""消息处理器 - 协调消息处理流程"""
from typing import Dict, Any
from loguru import logger
from core.message_filter import MessageFilter
from utils.security import security_limiter
from utils.message_logger import (
    log_received_message,
    log_agent_processing,
    log_agent_response,
    log_message_sent,
    log_security_block
)


class MessageHandler:
    """消息处理器 - 协调过滤、Agent 处理和响应发送"""

    def __init__(self, agent, napcat_client, message_filter: MessageFilter):
        """
        初始化消息处理器

        Args:
            agent: LangGraph Agent 实例
            napcat_client: NapCat 客户端实例
            message_filter: 消息过滤器实例
        """
        self.agent = agent
        self.napcat_client = napcat_client
        self.message_filter = message_filter

    async def handle_message(self, message_data: Dict[str, Any]):
        """
        处理消息

        Args:
            message_data: NapCat 消息数据
        """
        try:
            # 检查是否需要响应
            if not self.message_filter.should_respond(message_data):
                return

            # 提取消息信息
            group_id = message_data.get("group_id")
            user_id = message_data.get("user_id")
            sender_name = message_data.get("sender", {}).get("nickname", "未知")

            # 提取并清理消息文本
            raw_message = self.message_filter._extract_message_text(message_data)
            clean_text = self.message_filter.clean_message(raw_message)

            if not clean_text:
                logger.debug("消息清理后为空，跳过处理")
                return

            # 记录收到的有效消息
            log_received_message(group_id, user_id, sender_name, clean_text)

            logger.info(f"处理消息 - 群:{group_id}, 用户:{sender_name}({user_id}), 内容:{clean_text[:50]}")

            # 生成 thread_id（用于对话记忆）
            thread_id = f"group_{group_id}_user_{user_id}"

            # 记录 Agent 开始处理
            log_agent_processing(group_id, user_id, clean_text)

            # 调用 Agent 生成回复
            response = await self.agent.process_message(clean_text, thread_id)

            if not response:
                logger.warning("Agent 未生成有效回复")
                return

            # 记录 Agent 生成的回复
            log_agent_response(group_id, user_id, response)

            # 安全检查
            if not security_limiter.check_all(group_id, response):
                log_security_block(group_id, "消息长度或频率超限")
                logger.warning(f"消息未通过安全检查，不发送: {response[:50]}")
                return

            # 发送回复
            success = await self.napcat_client.send_group_msg(group_id, response)

            # 记录发送结果
            log_message_sent(group_id, success)

            if success:
                logger.success(f"回复发送成功 -> 群 {group_id}")
            else:
                logger.error(f"回复发送失败 -> 群 {group_id}")

        except Exception as e:
            logger.error(f"处理消息异常: {e}", exc_info=True)
