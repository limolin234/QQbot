"""Forward 工作流：使用 LangGraph 判定消息是否值得转发。"""

from __future__ import annotations

from typing import Any, TypedDict
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .agent_config_loader import load_current_agent_config

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


FORWARD_AGENT_CONFIG = load_current_agent_config(__file__)


class ForwardDecision(BaseModel):
    should_forward: bool = Field(description="是否需要转发给主人")
    reason: str = Field(default="", description="转发判定理由")


class ForwardState(TypedDict):
    ts: str
    group_id: str
    user_id: str
    user_name: str
    cleaned_message: str
    should_forward: bool
    reason: str


def get_forward_monitor_group_ids() -> set[str]:
    monitor_group_ids = FORWARD_AGENT_CONFIG.get("monitor_group_qq_number", [])
    if not isinstance(monitor_group_ids, list):
        return set()
    return {str(group_id).strip() for group_id in monitor_group_ids if str(group_id).strip()}


def run_forward_graph(
    *,
    ts: str,
    group_id: str,
    user_id: str,
    user_name: str,
    cleaned_message: str,
) -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(override=False)

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("缺少 LLM_API_KEY，请在 .env 中配置")

    base_url = os.getenv("LLM_API_BASE_URL")
    model_name = str(FORWARD_AGENT_CONFIG.get("model") or "gpt-4o-mini")
    try:
        temperature = float(FORWARD_AGENT_CONFIG.get("temperature", 0.0))
    except (TypeError, ValueError):
        temperature = 0.0

    llm_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "temperature": temperature,
        "openai_api_key": api_key,
    }
    if base_url:
        llm_kwargs["openai_api_base"] = base_url

    llm = ChatOpenAI(**llm_kwargs).with_structured_output(ForwardDecision)
    decision_prompt = str(
        FORWARD_AGENT_CONFIG.get("forward_decision_prompt")
        or "你是消息转发判定助手。仅返回 JSON：{\"should_forward\":true/false,\"reason\":\"...\"}"
    )

    def decide_node(state: ForwardState) -> ForwardState:
        llm_input = {
            "ts": state["ts"],
            "group_id": state["group_id"],
            "user_id": state["user_id"],
            "user_name": state["user_name"],
            "cleaned_message": state["cleaned_message"],
        }
        result = llm.invoke(
            [
                SystemMessage(content=decision_prompt),
                HumanMessage(content=json.dumps(llm_input, ensure_ascii=False)),
            ]
        )
        return {
            **state,
            "should_forward": bool(result.should_forward),
            "reason": (result.reason or "").strip(),
        }

    graph = StateGraph(ForwardState)
    graph.add_node("decide", decide_node)
    graph.add_edge(START, "decide")
    graph.add_edge("decide", END)
    app = graph.compile()

    final_state = app.invoke(
        {
            "ts": ts,
            "group_id": str(group_id),
            "user_id": str(user_id),
            "user_name": str(user_name),
            "cleaned_message": cleaned_message,
            "should_forward": False,
            "reason": "",
        }
    )

    should_forward = bool(final_state.get("should_forward", False))
    reason = str(final_state.get("reason", "")).strip()
    forward_text = (
        "[FORWARD]\n"
        f"时间: {ts}\n"
        f"群号: {group_id}\n"
        f"发送者: {user_name}({user_id})\n"
        f"内容: {cleaned_message}\n"
        f"判定: {reason or '命中转发规则'}"
    )
    return {
        "should_forward": should_forward,
        "reason": reason,
        "forward_text": forward_text,
    }

