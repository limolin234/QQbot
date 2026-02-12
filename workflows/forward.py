"""Forward 工作流：使用 LangGraph 判定消息是否值得转发。"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, TypedDict
import json
import os
import re

from ncatbot.core import GroupMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent_pool import submit_agent_job
from bot import QQnumber, bot
from .agent_observe import bind_agent_event, generate_run_id
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


def clean_message(raw_message: str) -> str:
    at_matches = re.findall(r"\[CQ:at,qq=(\d+)\]", raw_message)
    at_text = f"[@{','.join(at_matches)}]" if at_matches else ""
    cq_patterns = {
        r"\[CQ:image,file=[^]]+\]": "[图片]",
        r"\[CQ:reply,id=\d+\]": "[回复]",
        r"\[CQ:file,file=[^]]+\]": "[文件]",
        r"\[CQ:record,file=[^]]+\]": "[语音]",
        r"\[CQ:video,file=[^]]+\]": "[视频]",
    }
    cleaned = raw_message
    for pattern, replacement in cq_patterns.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    if at_matches:
        cleaned = re.sub(r"\[CQ:at,qq=\d+\]", at_text, cleaned)
    return cleaned.strip()


def _extract_message_ts(msg: GroupMessage) -> str:
    ts_candidate = getattr(msg, "time", None)
    if ts_candidate is not None:
        try:
            ts_value = float(ts_candidate)
            from datetime import datetime, timezone

            return datetime.fromtimestamp(ts_value, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        except (TypeError, ValueError, OSError):
            pass
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _extract_user_name(msg: GroupMessage) -> str:
    sender = getattr(msg, "sender", None)

    def pick(data, *keys):
        for key in keys:
            if isinstance(data, dict):
                value = data.get(key)
            else:
                value = getattr(data, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    for source in (msg, sender):
        if not source:
            continue
        name = pick(
            source,
            "card",
            "group_card",
            "display_name",
            "nickname",
            "nick",
            "user_name",
            "sender_nickname",
        )
        if name:
            return name

    return str(getattr(msg, "user_id", "unknown_user"))


async def enqueue_forward_by_monitor_group(msg: GroupMessage) -> bool:
    monitor_group_ids = get_forward_monitor_group_ids()
    group_id = str(getattr(msg, "group_id", ""))
    if group_id not in monitor_group_ids:
        return False

    cleaned_message = clean_message(getattr(msg, "raw_message", "") or "")
    ts = _extract_message_ts(msg)
    user_id = str(getattr(msg, "user_id", "unknown"))
    user_name = _extract_user_name(msg)

    task = asyncio.create_task(
        _execute_forward(
            ts=ts,
            group_id=group_id,
            user_id=user_id,
            user_name=user_name,
            cleaned_message=cleaned_message,
        )
    )

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except Exception as error:
            print(f"[FORWARD-ASYNC-ERROR] group={group_id} user={user_id} error={error}")

    task.add_done_callback(_on_done)
    return True


async def _execute_forward(
    *,
    ts: str,
    group_id: str,
    user_id: str,
    user_name: str,
    cleaned_message: str,
) -> None:

    run_id = generate_run_id()
    started = perf_counter()
    log_event = bind_agent_event(
        agent_name="forward",
        task_type="FORWARD",
        run_id=run_id,
        chat_type="group",
        group_id=group_id,
        user_id=user_id,
        user_name=user_name,
        ts=ts,
    )
    log_event(stage="start", extra={"cleaned_message_chars": len(cleaned_message)})

    try:
        result = await submit_agent_job(
            run_forward_graph,
            ts=ts,
            group_id=group_id,
            user_id=user_id,
            user_name=user_name,
            cleaned_message=cleaned_message,
            priority=5,
            timeout=120.0,
            run_in_thread=True,
        )
        if result.get("should_forward"):
            await bot.api.post_private_msg(QQnumber, text=str(result.get("forward_text", cleaned_message)))
        elapsed_ms = (perf_counter() - started) * 1000
        log_event(
            stage="end",
            latency_ms=elapsed_ms,
            decision={
                "should_forward": bool(result.get("should_forward")),
                "reason": str(result.get("reason", "")),
                "forward_text_chars": len(str(result.get("forward_text", "") or "")),
            },
        )
        print(
            "[FORWARD] "
            f"group={group_id} user={user_id} should_forward={bool(result.get('should_forward'))} "
            f"reason={result.get('reason', '')}"
        )
    except Exception as error:
        elapsed_ms = (perf_counter() - started) * 1000
        log_event(stage="error", latency_ms=elapsed_ms, error=str(error))
        print(f"[FORWARD-ERROR] group={group_id} user={user_id} error={error}")
    return True


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
