"""通知摘要 Agent - 识别重要通知并发送私聊摘要"""
import re
from typing import Optional, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from loguru import logger
from utils.message_logger import log_agent_processing, log_private_message_sent
from core.agent_factory import AgentFactory


# 定义通知分析结果的 Pydantic 模型
class NotificationAnalysis(BaseModel):
    """通知分析结果"""
    is_important: bool = Field(
        description="是否为重要通知"
    )
    category: Optional[str] = Field(
        default="",
        description="通知类别，如：作业、考试、活动等"
    )
    summary: Optional[str] = Field(
        default="",
        description="简洁的摘要（50字以内）"
    )
    key_info: Optional[str] = Field(
        default="",
        description="关键信息（时间、地点等）"
    )

    model_config = {
        "populate_by_name": True  # Pydantic v2 语法
    }


# 定义自定义 State
class NotificationState(TypedDict):
    """通知分析状态"""
    message: str  # 原始消息
    is_important: bool  # 是否重要
    category: str  # 类别
    summary: str  # 摘要
    key_info: str  # 关键信息


@AgentFactory.register("NotificationAgent")
class NotificationAgent:
    """通知摘要 Agent - 使用 LangGraph 识别重要通知并发送摘要"""

    def __init__(self, agent_id: str, config: dict, api_key: str, base_url: str, napcat_client):
        """
        初始化通知摘要 Agent

        Args:
            agent_id: Agent ID
            config: Agent 配置
            api_key: API Key
            base_url: API Base URL
            napcat_client: NapCat 客户端实例
        """
        self.agent_id = agent_id
        self.agent_name = config.get("name", "通知摘要助手")
        self.config = config
        self.napcat_client = napcat_client

        # 配置参数
        self.model = config.get("model", "deepseek-v3")
        self.monitored_groups = config.get("monitored_groups", [])
        self.target_user = config.get("target_user")
        self.trigger_mode = config.get("trigger_mode", "all")
        self.keywords = config.get("keywords", [])
        self.notification_prompt = config.get("notification_prompt", "")
        self.summary_prompt = config.get("summary_prompt", "")

        # 初始化 LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=self.model,
            temperature=0.3,  # 较低温度以获得更稳定的判断
            model_kwargs={"response_format": {"type": "json_object"}}  # 强制 JSON 输出
        )

        # 构建 LangGraph（通知识别不需要记忆功能）
        self.graph = self._build_graph()

        # 统计信息
        self.stats = {
            "total_processed": 0,
            "total_time": 0.0,
            "last_run": None,
            "success": 0,
            "errors": 0,
            "important_notifications": 0,
            "sent_summaries": 0
        }

        # 启用/禁用标志
        self.enabled = config.get("enabled", True)

        logger.info(f"NotificationAgent 初始化完成: {self.agent_name}")

    def _build_graph(self):
        """构建 LangGraph 工作流"""
        # 创建状态图（使用自定义 State）
        graph_builder = StateGraph(NotificationState)

        # 添加节点
        graph_builder.add_node("analyze", self._analyze_node)

        # 添加边
        graph_builder.add_edge(START, "analyze")
        graph_builder.add_edge("analyze", END)

        # 编译图
        graph = graph_builder.compile()

        logger.debug("NotificationAgent LangGraph 工作流构建完成")
        return graph

    def _analyze_node(self, state: NotificationState):
        """分析节点：判断是否为重要通知"""
        user_message = state["message"]

        # 构建分析提示（强调 JSON 格式）
        system_prompt = """你是一个通知识别助手。你的任务是判断消息是否为重要通知，并以 JSON 格式返回结果。

重要通知包括：
- 作业通知（布置作业、作业截止日期）
- 考试安排（考试时间、地点、科目）
- 课程变更（调课、停课、补课）
- 重要活动通知（讲座、会议、活动）
- 截止日期提醒（报名、提交材料等）
- 成绩公布
- 学校通知
- 荣誉奖项相关通知

不重要的消息：
- 日常闲聊
- 问候语
- 无关紧要的讨论

你必须返回以下格式的 JSON（不要添加任何其他文字）：
{
  "is_important": true 或 false,
  "category": "通知类别",
  "summary": "简洁的摘要（50字以内）",
  "key_info": "关键信息（时间、地点等）"
}

如果不是重要通知，返回：
{
  "is_important": false,
  "category": "",
  "summary": "",
  "key_info": ""
}"""

        try:
            # 调用 LLM
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"请分析以下消息并返回 JSON：\n\n{user_message}")
            ])

            logger.debug(f"LLM 原始响应: {response.content}")

            # 清理响应内容
            content = response.content.strip()

            # 移除可能的 markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]

            if content.endswith("```"):
                content = content[:-3]

            content = content.strip()

            # 解析 JSON
            import json
            result = json.loads(content)

            logger.info(f"✅ 通知分析成功: is_important={result.get('is_important')}, category={result.get('category')}")

            # 返回更新后的状态
            return {
                "message": user_message,
                "is_important": result.get("is_important", False),
                "category": result.get("category", ""),
                "summary": result.get("summary", ""),
                "key_info": result.get("key_info", "")
            }

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 解析失败: {e}")
            logger.error(f"LLM 响应内容: {response.content}")

            # 不使用后备方案，直接返回失败
            return {
                "message": user_message,
                "is_important": False,
                "category": "",
                "summary": "",
                "key_info": ""
            }

        except Exception as e:
            logger.error(f"❌ 分析节点异常: {e}", exc_info=True)

            # 不使用后备方案，直接返回失败
            return {
                "message": user_message,
                "is_important": False,
                "category": "",
                "summary": "",
                "key_info": ""
            }

    def _should_summarize(self, state):
        """条件判断：是否需要生成摘要"""
        return "summarize" if state.get("is_important") else "end"

    def _summarize_node(self, state):
        """摘要节点：生成详细摘要（可选，如果 analyze 已经生成了摘要则跳过）"""
        # 如果 analyze 节点已经生成了摘要，直接返回
        if state.get("summary"):
            return state

        # 否则使用 summary_prompt 生成更详细的摘要
        messages = state["messages"]
        user_message = messages[-1].content

        prompt = self.summary_prompt.format(message=user_message)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return {
                **state,
                "summary": response.content
            }
        except Exception as e:
            logger.error(f"摘要节点异常: {e}")
            return state

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

        # 根据触发模式判断
        if self.trigger_mode == "all":
            # 全部监听模式（年级通知群）
            return True

        elif self.trigger_mode == "keywords":
            # 关键词模式
            message_text = self._extract_message_text(message_data)
            if not self.keywords:
                return True  # 如果没有配置关键词，则全部监听
            return any(re.search(kw, message_text, re.IGNORECASE) for kw in self.keywords)

        elif self.trigger_mode == "hybrid":
            # 混合模式（先关键词过滤）
            message_text = self._extract_message_text(message_data)
            if self.keywords:
                return any(re.search(kw, message_text, re.IGNORECASE) for kw in self.keywords)
            return True  # 如果没有配置关键词，则全部监听

        return False

    async def process_message(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理消息

        Args:
            message_data: NapCat 消息数据

        Returns:
            响应动作字典
        """
        try:
            self.stats["total_processed"] += 1

            # 提取消息文本
            message_text = self._extract_message_text(message_data)
            group_id = message_data.get("group_id")
            user_id = message_data.get("user_id")
            sender = message_data.get("sender", {}).get("nickname", "未知")

            # 记录 Agent 开始处理
            log_agent_processing(group_id, user_id, message_text[:50], self.agent_name)

            logger.debug(f"NotificationAgent 处理消息: {message_text[:50]}")

            # 调用 graph
            result = self.graph.invoke({
                "message": message_text,
                "is_important": False,
                "category": "",
                "summary": "",
                "key_info": ""
            })

            logger.info(f"📊 Graph 返回结果: {result}")

            # 如果是重要通知，发送私聊
            if result.get("is_important"):
                logger.info(f"🔔 检测到重要通知，准备发送私聊")
                self.stats["important_notifications"] += 1

                category = result.get("category", "通知")
                summary = result.get("summary", "")
                key_info = result.get("key_info", "")

                # 构建私聊消息
                private_msg = f"""📢 重要通知提醒

【{category}】
群号：{group_id}
发送者：{sender}

📝 摘要：
{summary}

{f"⚠️ 关键信息：{key_info}" if key_info else ""}

---
原始消息：
{message_text[:200]}{"..." if len(message_text) > 200 else ""}"""

                # 发送私聊
                if self.target_user:
                    success = await self.napcat_client.send_private_msg(
                        self.target_user,
                        private_msg
                    )

                    # 记录发送结果
                    log_private_message_sent(self.target_user, success, self.agent_name)

                    if success:
                        self.stats["sent_summaries"] += 1
                        logger.success(f"通知摘要已发送给用户 {self.target_user}")
                    else:
                        self.stats["errors"] += 1
                        logger.error(f"发送通知摘要失败")

                    return {
                        "action": "send_private",
                        "target": self.target_user,
                        "message": private_msg,
                        "success": success
                    }
                else:
                    logger.warning("未配置 target_user，无法发送私聊")
                    return None

            return None

        except Exception as e:
            logger.error(f"NotificationAgent 处理失败: {e}", exc_info=True)
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

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            **self.stats
        }
