"""简单的 LangGraph 聊天 Agent"""
import re
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger
from utils.message_logger import log_agent_processing, log_agent_response, log_message_sent
from core.agent_factory import AgentFactory


@AgentFactory.register("SimpleChatAgent")
class SimpleChatAgent:
    """简单的聊天 Agent，使用 LangGraph 管理对话状态"""

    def __init__(self, agent_id: str, config: dict, api_key: str, base_url: str, napcat_client):
        """
        初始化 Agent

        Args:
            agent_id: Agent ID
            config: Agent 配置
            api_key: OpenAI API Key
            base_url: API Base URL
            napcat_client: NapCat 客户端实例
        """
        self.agent_id = agent_id
        self.agent_name = config.get("name", "聊天助手")
        self.config = config
        self.napcat_client = napcat_client

        self.api_key = api_key
        self.base_url = base_url
        self.model = config.get("model", "deepseek-v3")
        self.system_prompt = config.get("system_prompt", "你是一个友好的助手")

        # 触发配置
        self.monitored_groups = config.get("monitored_groups", [])
        self.trigger_mode = config.get("trigger_mode", "hybrid")
        self.wake_words = config.get("wake_words", [])
        self.require_at = config.get("require_at", False)

        # 初始化 LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=self.model,
            temperature=0.7
        )

        # 初始化记忆存储
        self.memory = MemorySaver()

        # 构建 LangGraph
        self.graph = self._build_graph()

        # 统计信息
        self.stats = {
            "total_processed": 0,
            "total_time": 0.0,
            "last_run": None,
            "success": 0,
            "errors": 0
        }

        # 启用/禁用标志
        self.enabled = config.get("enabled", True)

        logger.info(f"SimpleChatAgent 初始化完成: {self.agent_name}, 模型: {self.model}")
        logger.debug(f"系统提示词: {self.system_prompt[:50]}...")

    def _build_graph(self):
        """构建 LangGraph 工作流"""
        # 创建状态图
        graph_builder = StateGraph(MessagesState)

        # 添加聊天节点
        graph_builder.add_node("chatbot", self._chatbot_node)

        # 设置边
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        # 编译图，启用记忆功能
        graph = graph_builder.compile(checkpointer=self.memory)

        logger.debug("LangGraph 工作流构建完成")
        return graph

    def _chatbot_node(self, state: MessagesState):
        """
        聊天节点 - 调用 LLM 生成回复

        Args:
            state: 当前对话状态

        Returns:
            更新后的状态
        """
        try:
            # 构建消息列表，添加系统提示词
            messages = state["messages"]

            # 如果是第一条消息，添加系统提示词
            if len(messages) == 1:
                messages = [SystemMessage(content=self.system_prompt)] + messages

            # 调用 LLM
            response = self.llm.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            # 返回错误消息
            return {"messages": [{"role": "assistant", "content": "抱歉，我遇到了一些问题，请稍后再试。"}]}

    def should_trigger(self, message_data: Dict[str, Any]) -> bool:
        """
        判断是否应该触发此 Agent

        Args:
            message_data: NapCat 消息数据

        Returns:
            True 如果应该触发
        """
        # 检查消息类型
        if message_data.get("post_type") != "message":
            return False
        if message_data.get("message_type") != "group":
            return False

        # 检查群号
        group_id = message_data.get("group_id")
        if group_id not in self.monitored_groups:
            return False

        # 提取消息文本
        message_text = self._extract_message_text(message_data)
        raw_message = message_data.get("message", "")

        # 获取 bot QQ 号
        from config.bot_config import BOT_QQ
        bot_qq = BOT_QQ

        # 检查是否 @ 了 bot
        is_at = False
        if bot_qq:
            at_pattern = f"[CQ:at,qq={bot_qq}]"
            is_at = at_pattern in str(raw_message)

        # 如果要求必须 @，则检查 @
        if self.require_at and not is_at:
            return False

        # 检查唤醒词
        has_wake_word = any(re.search(word, message_text, re.IGNORECASE) for word in self.wake_words)

        # 触发条件：@ 或包含唤醒词
        return is_at or has_wake_word

    async def process_message(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理用户消息并生成回复

        Args:
            message_data: NapCat 消息数据

        Returns:
            响应动作字典
        """
        try:
            self.stats["total_processed"] += 1

            # 提取消息信息
            group_id = message_data.get("group_id")
            user_id = message_data.get("user_id")

            # 提取并清理消息文本
            raw_message = message_data.get("message", "")
            clean_text = self._clean_message(str(raw_message))

            # 记录 Agent 开始处理
            log_agent_processing(group_id, user_id, clean_text, self.agent_name)

            # 生成 thread_id（用于对话记忆）
            thread_id = f"group_{group_id}_user_{user_id}"

            # 配置 thread_id 以启用记忆功能
            config = {"configurable": {"thread_id": thread_id}}

            # 调用 graph
            result = self.graph.invoke(
                {"messages": [{"role": "user", "content": clean_text}]},
                config
            )

            # 提取回复
            if result and "messages" in result and len(result["messages"]) > 0:
                reply = result["messages"][-1].content
                logger.debug(f"SimpleChatAgent 回复: {reply[:100]}")

                # 记录 Agent 回复
                log_agent_response(group_id, user_id, reply, self.agent_name)

                # 发送群消息
                success = await self.napcat_client.send_group_msg(group_id, reply)

                # 记录发送结果
                log_message_sent(group_id, success, self.agent_name)

                if success:
                    self.stats["success"] += 1
                else:
                    self.stats["errors"] += 1

                return {
                    "action": "send_group",
                    "target": group_id,
                    "message": reply,
                    "success": success
                }
            else:
                logger.warning("SimpleChatAgent 未返回有效回复")
                self.stats["errors"] += 1
                return None

        except Exception as e:
            logger.error(f"SimpleChatAgent 处理失败: {e}", exc_info=True)
            self.stats["errors"] += 1
            return None

    def _extract_message_text(self, message_data: Dict[str, Any]) -> str:
        """提取消息文本"""
        message = message_data.get("message", "")

        # 如果是字符串，直接返回
        if isinstance(message, str):
            return message

        # 如果是列表（消息段数组），提取文本
        if isinstance(message, list):
            text_parts = []
            for segment in message:
                if isinstance(segment, dict) and segment.get("type") == "text":
                    text_parts.append(segment.get("data", {}).get("text", ""))
            return "".join(text_parts)

        return ""

    def _clean_message(self, raw_message: str) -> str:
        """清理消息，移除 @ 和 CQ 码"""
        # 移除 @ 相关的 CQ 码
        text = re.sub(r'\[CQ:at,qq=\d+\]', '', raw_message)
        # 移除其他 CQ 码
        text = re.sub(r'\[CQ:[^\]]+\]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_conversation_state(self, thread_id: str):
        """
        获取对话状态（用于调试）

        Args:
            thread_id: 线程 ID

        Returns:
            对话状态快照
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self.graph.get_state(config)
            return snapshot
        except Exception as e:
            logger.error(f"获取对话状态失败: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            **self.stats
        }
