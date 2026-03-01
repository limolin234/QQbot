"""DidaAgent 工作流：当前只实现“是否需要回复”的判定。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Literal, Optional, TypedDict
import json
import os
import re

from ncatbot.core import GroupMessage, PrivateMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent_pool import submit_agent_job
from bot import bot
from ..agent_observe import bind_agent_event, generate_run_id
from ..agent_config_loader import load_current_agent_config
from .dida_scheduler import dida_scheduler

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


DIDA_AGENT_CONFIG = load_current_agent_config(__file__)


def get_dida_agent_runtime_config() -> dict[str, Any]:
    """读取并规范化 dida_agent 运行时配置。"""

    def int_config(name: str, default: int) -> int:
        try:
            value = int(DIDA_AGENT_CONFIG.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(value, 0)

    def bool_config(name: str, default: bool) -> bool:
        value = DIDA_AGENT_CONFIG.get(name, default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    admin_qqs_raw = DIDA_AGENT_CONFIG.get("admin_qqs", [])
    if isinstance(admin_qqs_raw, str):
        admin_qqs = [x.strip() for x in admin_qqs_raw.split(",") if x.strip()]
    elif isinstance(admin_qqs_raw, list):
        admin_qqs = [str(x).strip() for x in admin_qqs_raw if str(x).strip()]
    else:
        admin_qqs = []

    return {
        "context_history_limit": int_config("context_history_limit", 12),
        "context_max_chars": int_config("context_max_chars", 1800),
        "context_window_seconds": int_config("context_window_seconds", 1800),
        "min_reply_interval_seconds": int_config("min_reply_interval_seconds", 300),
        "flush_check_interval_seconds": int_config("flush_check_interval_seconds", 10),
        "pending_expire_seconds": int_config("pending_expire_seconds", 3600),
        "pending_max_messages": int_config("pending_max_messages", 50),
        "bypass_cooldown_when_at_bot": bool_config("bypass_cooldown_when_at_bot", True),
        "bot_qq": str(DIDA_AGENT_CONFIG.get("bot_qq", "")).strip(),
        "admin_qqs": admin_qqs,
    }


def _parse_timestamp_to_epoch_seconds(ts_text: str) -> float | None:
    text = str(ts_text).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    iso_text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def get_dida_agent_monitor_numbers(chat_type: str = "group") -> set[str]:
    """读取 dida_agent 配置里启用规则的监控号码集合。"""
    rules = DIDA_AGENT_CONFIG.get("rules", [])
    if not isinstance(rules, list):
        return set()
    target_chat_type = str(chat_type).strip().lower()
    monitor_numbers: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict) or not bool(rule.get("enabled", False)):
            continue
        if str(rule.get("chat_type", "")).strip().lower() != target_chat_type:
            continue
        number = str(rule.get("number", "")).strip()
        if number:
            monitor_numbers.add(number)
    return monitor_numbers


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _strip_markdown_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw.startswith("```"):
        return raw
    fenced_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw, flags=re.IGNORECASE)
    if fenced_match:
        return str(fenced_match.group(1) or "").strip()
    lines = raw.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return raw


def _extract_reply_text_from_raw_output(raw_output: Any) -> str:
    raw_text = ""
    if hasattr(raw_output, "content"):
        raw_text = _content_to_text(getattr(raw_output, "content"))
    elif isinstance(raw_output, dict):
        reply_text = str(raw_output.get("reply_text", "")).strip()
        if reply_text:
            return reply_text
        raw_text = _content_to_text(raw_output.get("content", ""))
    else:
        raw_text = str(raw_output or "").strip()

    cleaned = _strip_markdown_code_fence(raw_text)
    if not cleaned:
        return ""

    candidates = [cleaned]
    json_obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_obj_match:
        obj_text = str(json_obj_match.group(0) or "").strip()
        if obj_text and obj_text != cleaned:
            candidates.insert(0, obj_text)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            reply_text = str(obj.get("reply_text", "")).strip()
            if reply_text:
                return reply_text

    if cleaned.startswith("{") or cleaned.startswith("["):
        return ""
    return cleaned


def _default_reply_when_parse_failed() -> str:
    return "抱歉，刚才回复格式异常，请再说一次，我马上继续。"


class DidaAgentDispatcher:
    """DidaAgent 接入层：监控命中 + 冷却窗口 + pending 聚合/过期。"""

    def __init__(self):
        self._state_lock = asyncio.Lock()
        self._session_states: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _build_session_key(chat_type: str, monitor_value: str) -> str:
        return f"{chat_type}:{monitor_value}"

    @staticmethod
    def _build_payload(
        *,
        raw_message: str,
        cleaned_message: str,
        chat_type: str,
        group_id: str,
        user_id: str,
        user_name: str,
        ts: str,
    ) -> dict[str, str]:
        return {
            "raw_message": raw_message,
            "cleaned_message": cleaned_message,
            "chat_type": chat_type,
            "group_id": group_id,
            "user_id": user_id,
            "user_name": user_name,
            "ts": ts,
        }

    @staticmethod
    def payload_to_message(payload: dict[str, str]) -> SimpleNamespace:
        return SimpleNamespace(
            raw_message=str(payload.get("raw_message", "")),
            cleaned_message=str(payload.get("cleaned_message", "")),
            chat_type=str(payload.get("chat_type", "")),
            group_id=str(payload.get("group_id", "")),
            user_id=str(payload.get("user_id", "unknown")),
            user_name=str(payload.get("user_name", "")),
            ts=str(payload.get("ts", "")),
        )

    @staticmethod
    def _hit_at_bot(raw_message: str, bot_qq: str) -> bool:
        at_ids = re.findall(r"\[CQ:at,qq=(\d+)\]", raw_message or "")
        if not at_ids:
            return False
        configured_bot_qq = str(bot_qq).strip()
        if not configured_bot_qq:
            return False
        return configured_bot_qq in at_ids

    async def enqueue_if_monitored(
        self,
        *,
        chat_type: str,
        monitor_value: str,
        raw_message: str,
        cleaned_message: str,
        user_id: str,
        user_name: str,
        ts: str,
        enqueue_payload: Callable[[dict[str, str]], Awaitable[None]],
    ) -> bool:
        target_chat_type = str(chat_type).strip().lower()
        if target_chat_type not in {"group", "private"}:
            return False

        normalized_monitor_value = str(monitor_value)
        monitor_numbers = get_dida_agent_monitor_numbers(chat_type=target_chat_type)
        if normalized_monitor_value not in monitor_numbers:
            return False

        runtime_config = get_dida_agent_runtime_config()
        min_reply_interval = max(int(runtime_config.get("min_reply_interval_seconds", 300)), 0)
        pending_max_messages = max(int(runtime_config.get("pending_max_messages", 50)), 1)

        dida_agent_payload = self._build_payload(
            raw_message=str(raw_message),
            cleaned_message=str(cleaned_message),
            chat_type=target_chat_type,
            group_id=normalized_monitor_value if target_chat_type == "group" else "private",
            user_id=str(user_id),
            user_name=str(user_name),
            ts=str(ts),
        )
        log_event = bind_agent_event(
            agent_name="dida_agent",
            task_type="DIDA_AGENT",
            chat_type=target_chat_type,
            group_id=dida_agent_payload["group_id"],
            user_id=dida_agent_payload["user_id"],
            user_name=dida_agent_payload["user_name"],
            ts=dida_agent_payload["ts"],
        )

        bypass_cooldown = (
            target_chat_type == "group"
            and bool(runtime_config.get("bypass_cooldown_when_at_bot", True))
            and self._hit_at_bot(str(raw_message), str(runtime_config.get("bot_qq", "")))
        )
        if bypass_cooldown:
            log_event(
                stage="throttle_bypass_cooldown",
                extra={
                    "session": f"{target_chat_type}:{normalized_monitor_value}",
                    "reason": "at_bot",
                },
            )

        enqueue_now = True
        if min_reply_interval > 0 and not bypass_cooldown:
            session_key = self._build_session_key(target_chat_type, normalized_monitor_value)
            now = datetime.now().timestamp()
            async with self._state_lock:
                state = self._session_states.setdefault(
                    session_key,
                    {
                        "next_allowed_at": 0.0,
                        "pending_payload": None,
                        "pending_since": 0.0,
                        "pending_count": 0,
                    },
                )
                next_allowed_at = float(state.get("next_allowed_at", 0.0) or 0.0)
                if now < next_allowed_at:
                    state["pending_payload"] = dida_agent_payload
                    previous_count = int(state.get("pending_count", 0) or 0)
                    state["pending_count"] = min(previous_count + 1, pending_max_messages)
                    if not float(state.get("pending_since", 0.0) or 0.0):
                        state["pending_since"] = now
                    log_event(
                        stage="throttle_queued_pending",
                        extra={
                            "session": session_key,
                            "pending_count": int(state.get("pending_count", 0) or 0),
                            "cooldown_left_seconds": max(int(next_allowed_at - now), 0),
                        },
                    )
                    enqueue_now = False
                else:
                    state["next_allowed_at"] = now + min_reply_interval
                    state["pending_payload"] = None
                    state["pending_since"] = 0.0
                    state["pending_count"] = 0

        if enqueue_now:
            log_event(
                stage="throttle_enqueue_now",
                extra={
                    "session": f"{target_chat_type}:{normalized_monitor_value}",
                    "cooldown_seconds": min_reply_interval,
                },
            )
            await enqueue_payload(dida_agent_payload)
        return True

    async def pending_worker(self, *, enqueue_payload: Callable[[dict[str, str]], Awaitable[None]]) -> None:
        while True:
            runtime_config = get_dida_agent_runtime_config()
            check_interval = max(int(runtime_config.get("flush_check_interval_seconds", 10)), 1)
            min_interval = max(int(runtime_config.get("min_reply_interval_seconds", 300)), 0)
            pending_expire = max(int(runtime_config.get("pending_expire_seconds", 3600)), 0)
            now = datetime.now().timestamp()
            due_payloads: list[dict[str, str]] = []

            async with self._state_lock:
                for session_key, state in self._session_states.items():
                    pending_payload = state.get("pending_payload")
                    if not isinstance(pending_payload, dict):
                        continue

                    pending_event = bind_agent_event(
                        agent_name="dida_agent",
                        task_type="DIDA_AGENT",
                        chat_type=str(pending_payload.get("chat_type", "")),
                        group_id=str(pending_payload.get("group_id", "")),
                        user_id=str(pending_payload.get("user_id", "")),
                        user_name=str(pending_payload.get("user_name", "")),
                        ts=str(pending_payload.get("ts", "")),
                    )

                    pending_since = float(state.get("pending_since", now) or now)
                    if pending_expire > 0 and (now - pending_since) > pending_expire:
                        pending_event(
                            stage="throttle_pending_expired",
                            extra={
                                "session": session_key,
                                "pending_age_seconds": int(now - pending_since),
                            },
                        )
                        state["pending_payload"] = None
                        state["pending_since"] = 0.0
                        state["pending_count"] = 0
                        continue

                    next_allowed_at = float(state.get("next_allowed_at", 0.0) or 0.0)
                    if now < next_allowed_at:
                        continue

                    pending_event(
                        stage="throttle_pending_flush",
                        extra={
                            "session": session_key,
                            "pending_count": int(state.get("pending_count", 0) or 0),
                        },
                    )
                    due_payloads.append(pending_payload)
                    state["pending_payload"] = None
                    state["pending_since"] = 0.0
                    state["pending_count"] = 0
                    state["next_allowed_at"] = now + min_interval if min_interval > 0 else 0.0

            for payload in due_payloads:
                await enqueue_payload(payload)

            await asyncio.sleep(check_interval)


_DIDA_AGENT_DISPATCHER = DidaAgentDispatcher()


def get_dida_agent_dispatcher() -> DidaAgentDispatcher:
    return _DIDA_AGENT_DISPATCHER


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


def _extract_message_ts(msg: GroupMessage | PrivateMessage) -> str:
    ts_candidate = getattr(msg, "time", None)
    if ts_candidate is not None:
        try:
            ts_value = float(ts_candidate)
            return datetime.fromtimestamp(ts_value, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        except (TypeError, ValueError, OSError):
            pass
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _extract_user_name(msg: GroupMessage | PrivateMessage) -> str:
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


async def _execute_dida_agent_payload(payload: dict[str, str]) -> None:
    ts = str(payload.get("ts", ""))
    chat_type = str(payload.get("chat_type", ""))
    group_id = str(payload.get("group_id", ""))
    user_id = str(payload.get("user_id", ""))
    user_name = str(payload.get("user_name", ""))
    raw_message = str(payload.get("raw_message", ""))
    cleaned_message = str(payload.get("cleaned_message", raw_message))

    run_id = generate_run_id()
    started = perf_counter()
    log_event = bind_agent_event(
        agent_name="dida_agent",
        task_type="DIDA_AGENT",
        run_id=run_id,
        chat_type=chat_type,
        group_id=group_id,
        user_id=user_id,
        user_name=user_name,
        ts=ts,
    )
    log_event(stage="start")

    try:
        result = await submit_agent_job(
            run_dida_agent_pipeline,
            chat_type=chat_type,
            group_id=group_id,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            raw_message=raw_message,
            cleaned_message=cleaned_message,
            run_id=run_id,
            priority=0,
            timeout=120.0,
            run_in_thread=True,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        log_event(
            stage="end",
            latency_ms=elapsed_ms,
            decision={
                "should_reply": bool(result.get("should_reply", False)),
                "reason": str(result.get("reason", "")),
                "matched_rule": result.get("matched_rule"),
                "trigger_mode": result.get("trigger_mode", ""),
                "reply_length": len(str(result.get("reply_text", "") or "")),
            },
        )
        should_reply_flag = bool(result.get("should_reply", False))
        reason_text = str(result.get("reason", ""))
        is_new_mode = bool(result.get("is_new_mode", False))
        
        reply_text = str(result.get("reply_text", "") or "").strip()
        dida_action = result.get("dida_action")
        dida_response = ""
        
        if is_new_mode and should_reply_flag:
             action_result = result.get("action_result")
             context = result.get("context")
             rule = result.get("rule")
             engine = result.get("engine")
             
             execution_result = None
             
             if action_result and action_result.need_clarification:
                 execution_result = {
                     "ok": False,
                     "message": str(action_result.clarification_question),
                     "need_clarification": True
                 }
             elif action_result and action_result.dida_action:
                 dida_action = action_result.dida_action
                 # Admin logic
                 runtime_config = get_dida_agent_runtime_config()
                 admin_qqs = runtime_config.get("admin_qqs", [])
                 if str(user_id) in admin_qqs:
                    target_user_id = str(getattr(dida_action, "target_user_id", "") or "").strip()
                    if not target_user_id:
                        at_matches = re.findall(r"\[CQ:at,qq=(\d+)\]", raw_message or "")
                        for uid in at_matches:
                            if uid and uid != str(user_id):
                                setattr(dida_action, "target_user_id", uid)
                                break
                        if at_matches and not str(getattr(dida_action, "target_user_id", "") or "").strip():
                            setattr(dida_action, "target_user_id", str(at_matches[0]))
                 try:
                     execution_result = await dida_scheduler.execute_action_structured(
                         action=dida_action,
                         chat_type=chat_type,
                         group_id=group_id,
                         user_id=user_id,
                         user_name=user_name,
                     )
                 except Exception as action_error:
                     execution_result = {"ok": False, "message": f"Dida 操作失败：{action_error}"}
             
             # Model B
             try:
                 reply_res = await submit_agent_job(
                     engine.generate_final_reply,
                     reply_prompt=str(rule.get("reply_prompt", "")),
                     context=context,
                     action_result=action_result,
                     execution_result=execution_result,
                     rule=rule,
                     priority=0,
                     run_in_thread=True
                 )
                 reply_text = reply_res.reply_text
                 
                 # Append execution result message if available (for fixed format output)
                 if execution_result and isinstance(execution_result, dict):
                     dida_msg = str(execution_result.get("message", "") or "").strip()
                     if dida_msg:
                         reply_text = f"{reply_text}\n{dida_msg}".strip()

             except Exception as reply_error:
                 reply_text = f"（生成回复失败：{reply_error}）"
                 if execution_result:
                     reply_text += f"\n执行结果：{execution_result.get('message')}"

        elif dida_action is not None:
            runtime_config = get_dida_agent_runtime_config()
            admin_qqs = runtime_config.get("admin_qqs", [])
            if str(user_id) in admin_qqs:
                target_user_id = str(getattr(dida_action, "target_user_id", "") or "").strip()
                if not target_user_id:
                    at_matches = re.findall(r"\[CQ:at,qq=(\d+)\]", raw_message or "")
                    for uid in at_matches:
                        if uid and uid != str(user_id):
                            setattr(dida_action, "target_user_id", uid)
                            break
                    if at_matches and not str(getattr(dida_action, "target_user_id", "") or "").strip():
                        setattr(dida_action, "target_user_id", str(at_matches[0]))
            try:
                dida_response = await dida_scheduler.execute_action(
                    action=dida_action,
                    chat_type=chat_type,
                    group_id=group_id,
                    user_id=user_id,
                    user_name=user_name,
                )
            except Exception as action_error:
                dida_response = f"⚠️ Dida 操作失败：{action_error}"
        final_reply = reply_text
        if dida_response:
            final_reply = f"{reply_text}\n{dida_response}".strip() if reply_text else dida_response.strip()

        if should_reply_flag and final_reply:
            log_event(stage="send_start", extra={"reply_length": len(final_reply)})
            try:
                if chat_type == "group":
                    await bot.api.post_group_msg(group_id, text=final_reply)
                else:
                    await bot.api.post_private_msg(user_id, text=final_reply)
                log_event(stage="send_end", decision={"sent": True, "reply_length": len(final_reply)})
            except Exception as send_error:
                log_event(stage="send_error", error=str(send_error))
                print(
                    "[DIDA_AGENT-SEND-ERROR] "
                    f"chat={chat_type} group={group_id} user={user_id} error={send_error}"
                )
        if should_reply_flag:
            log_event(stage="send_skip", decision={"sent": False, "reason": "reply_text_empty"})

        print(
            f"[DIDA_AGENT] chat={chat_type} group={group_id} user={user_name}({user_id})\n"
            f"Input: {cleaned_message!r}\n"
            f"should_reply={should_reply_flag}\n"
            f"Reply: {final_reply!r}\n"
            f"Reason: {reason_text}"
        )
    except asyncio.TimeoutError:
        elapsed_ms = (perf_counter() - started) * 1000
        log_event(stage="timeout", latency_ms=elapsed_ms, error="Request timed out")
        print(
            f"[DIDA_AGENT-ERROR] chat={chat_type} group={group_id} user={user_id}\n"
            f"Input: {cleaned_message!r}\n"
            "error=TimeoutError (request took > 120s)"
        )
    except Exception as error:
        elapsed_ms = (perf_counter() - started) * 1000
        log_event(stage="error", latency_ms=elapsed_ms, error=str(error))
        import traceback
        traceback.print_exc()
        print(
            f"[DIDA_AGENT-ERROR] chat={chat_type} group={group_id} user={user_id}\n"
            f"Input: {cleaned_message!r}\n"
            f"error={repr(error)}"
        )


def _create_dida_agent_task(payload: dict[str, str]) -> None:
    task = asyncio.create_task(_execute_dida_agent_payload(payload))

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except Exception as error:
            print(f"[DIDA_AGENT-ASYNC-ERROR] error={error}")

    task.add_done_callback(_on_done)


async def _enqueue_dida_agent_payload(payload: dict[str, str]) -> None:
    _create_dida_agent_task(payload)


async def dida_agent_pending_worker() -> None:
    dispatcher = get_dida_agent_dispatcher()
    await dispatcher.pending_worker(enqueue_payload=_enqueue_dida_agent_payload)


async def enqueue_dida_agent_if_monitored(
    msg: GroupMessage | PrivateMessage,
    chat_type: str,
) -> bool:
    target_chat_type = str(chat_type).strip().lower()
    monitor_value = (
        str(getattr(msg, "group_id", ""))
        if target_chat_type == "group"
        else str(getattr(msg, "user_id", ""))
    )
    raw_message = getattr(msg, "raw_message", "") or ""
    dispatcher = get_dida_agent_dispatcher()
    return await dispatcher.enqueue_if_monitored(
        chat_type=target_chat_type,
        monitor_value=monitor_value,
        raw_message=raw_message,
        cleaned_message=clean_message(raw_message),
        user_id=str(getattr(msg, "user_id", "unknown")),
        user_name=_extract_user_name(msg),
        ts=_extract_message_ts(msg),
        enqueue_payload=_enqueue_dida_agent_payload,
    )


def load_recent_context_messages(
    *,
    chat_type: str,
    group_id: str,
    user_id: str,
    current_ts: str,
    current_cleaned_message: str,
    limit: int,
    max_chars: int,
    window_seconds: int,
    log_path: str = "message.jsonl",
) -> list[str]:
    """从 message.jsonl 读取最近上下文消息（按 chat_type + 会话范围）。"""
    try:
        with open(log_path, "r", encoding="utf-8") as file:
            raw_lines = [line.strip() for line in file if line.strip()]
    except OSError:
        return []

    matched_records: list[tuple[str, str, str]] = []
    target_chat_type = str(chat_type).strip().lower()
    target_group = str(group_id)
    target_user = str(user_id)
    current_epoch = _parse_timestamp_to_epoch_seconds(current_ts)
    window_enabled = bool(window_seconds > 0 and current_epoch is not None)
    for raw_line in raw_lines:
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        payload_chat_type = str(payload.get("chat_type", "")).strip().lower()
        if payload_chat_type != target_chat_type:
            continue
        if target_chat_type == "group" and str(payload.get("group_id", "")) != target_group:
            continue
        if target_chat_type == "private" and str(payload.get("user_id", "")) != target_user:
            continue

        cleaned_message = str(payload.get("cleaned_message", "")).strip()
        if not cleaned_message:
            continue
        ts = str(payload.get("ts", ""))
        if window_enabled:
            payload_epoch = _parse_timestamp_to_epoch_seconds(ts)
            if payload_epoch is not None and payload_epoch < (current_epoch - window_seconds):
                continue
        sender_name = str(payload.get("user_name", "")).strip() or str(payload.get("user_id", ""))
        matched_records.append((ts, sender_name, cleaned_message))

    context_lines: list[str] = []
    total_chars = 0
    skipped_current = False
    for ts, sender_name, message_text in reversed(matched_records):
        if (
            not skipped_current
            and str(ts) == str(current_ts)
            and str(message_text) == str(current_cleaned_message)
        ):
            skipped_current = True
            continue

        line = f"[{ts}] {sender_name}: {message_text}" if ts else f"{sender_name}: {message_text}"
        if total_chars + len(line) > max_chars:
            break
        context_lines.append(line)
        total_chars += len(line)
        if len(context_lines) >= limit:
            break

    return list(reversed(context_lines))


@dataclass
class DidaAgentMessageContext:
    chat_type: str
    group_id: str
    user_id: str
    user_name: str
    ts: str
    raw_message: str
    cleaned_message: str
    history_messages: list[str] = field(default_factory=list)
    run_id: str = ""


class DidaAgentAIDecision(BaseModel):
    should_reply: bool = Field(description="是否需要回复")
    reason: str = Field(default="", description="判定理由")


class DidaAction(BaseModel):
    action_type: Literal["create", "list", "delete", "complete", "update"] = Field(description="Dida 动作类型")
    title: Optional[str] = Field(default=None, description="任务标题")
    due_date: Optional[str] = Field(default=None, description="到期时间")
    task_id: Optional[str] = Field(default=None, description="任务 ID")
    project_id: Optional[str] = Field(default=None, description="项目 ID")
    content: Optional[str] = Field(default=None, description="任务内容")
    desc: Optional[str] = Field(default=None, description="任务描述")
    is_all_day: Optional[bool] = Field(default=None, description="是否全天")
    time_zone: Optional[str] = Field(default=None, description="时区")
    repeat_flag: Optional[str] = Field(default=None, description="重复规则")
    reminders: Optional[list[str]] = Field(default=None, description="提醒数组")
    limit: Optional[int] = Field(default=None, description="列表数量")
    target_user_id: Optional[str] = Field(default=None, description="目标用户QQ号，仅限管理员操作他人任务时使用")


class ActionLLMResult(BaseModel):
    dida_action: Optional[DidaAction] = Field(default=None, description="Dida 动作，若无明确动作或有歧义则为 None")
    need_clarification: bool = Field(default=False, description="是否需要用户澄清（如存在同名任务）")
    clarification_question: Optional[str] = Field(default=None, description="澄清问题内容（自然语言）")


class ReplyResponse(BaseModel):
    reply_text: str = Field(description="最终回复给用户的文本")


class DidaAgentGeneratedReply(BaseModel):
    reply_text: str = Field(default="", description="自动回复文本")
    dida_action: Optional[DidaAction] = Field(default=None, description="Dida 动作")


class DidaAgentAIState(TypedDict):
    prompt: str
    chat_type: str
    group_id: str
    user_id: str
    user_name: str
    ts: str
    raw_message: str
    cleaned_message: str
    history_messages: list[str]
    should_reply: bool
    reason: str


class DidaAgentGenerateState(TypedDict):
    prompt: str
    chat_type: str
    group_id: str
    user_id: str
    user_name: str
    ts: str
    raw_message: str
    cleaned_message: str
    history_messages: list[str]
    reply_text: str
    dida_action: Optional[DidaAction]


class DidaAgentDecisionEngine:
    """自动回复判定器：负责读取规则表达式并计算 should_reply。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config if isinstance(config, dict) else {}
        raw_rules = self.config.get("rules", [])
        self.rules = [rule for rule in raw_rules if isinstance(rule, dict) and bool(rule.get("enabled", False))]
        
        # Validate rules for Dida config
        for idx, rule in enumerate(self.rules):
            if rule.get("dida_enabled"):
                action_prompt = str(rule.get("action_prompt") or "").strip()
                if not action_prompt:
                    chat_type = str(rule.get("chat_type", "")).strip()
                    number = str(rule.get("number", "")).strip()
                    print(
                        f"[DIDA_AGENT-CONFIG-ERROR] rule#{idx} (chat_type={chat_type} number={number}) "
                        "enabled dida but missing 'action_prompt'. Disabling dida for this rule."
                    )
                    rule["dida_enabled"] = False

        self.model = str(self.config.get("model") or "gpt-4o-mini")
        try:
            self.temperature = float(self.config.get("temperature", 0.0))
        except (TypeError, ValueError):
            self.temperature = 0.0

    def should_reply(self, context: DidaAgentMessageContext) -> dict[str, Any]:
        context_event = bind_agent_event(
            agent_name="dida_agent",
            task_type="DIDA_AGENT",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
        )
        matching_rules = []
        for rule in self.rules:
            rule_chat_type = str(rule.get("chat_type", "")).strip().lower()
            if rule_chat_type != context.chat_type:
                continue
            target_number = str(rule.get("number", "")).strip()
            current_number = context.group_id if context.chat_type == "group" else context.user_id
            if target_number != str(current_number):
                continue
            matching_rules.append(rule)

        context_event(
            stage="rules_filtered",
            extra={"configured_rules": len(self.rules), "matched_rules": len(matching_rules)},
        )

        if not matching_rules:
            result = {
                "should_reply": False,
                "reason": "未命中启用规则",
                "matched_rule": None,
            }
            context_event(stage="decision_end", decision=result)
            return result

        failure_reasons: list[str] = []
        for idx, rule in enumerate(matching_rules):
            expression = str(rule.get("trigger_mode") or "always").strip()
            ok, reason = self.evaluate_trigger_expression(expression, rule, context)
            context_event(
                stage="rule_evaluated",
                decision={"should_reply": ok, "reason": reason},
                extra={"rule_index": idx, "trigger_mode": expression},
            )
            if ok:
                result = {
                    "should_reply": True,
                    "reason": reason,
                    "matched_rule": idx,
                    "trigger_mode": expression,
                    "reply_prompt": str(rule.get("reply_prompt") or "").strip(),
                    "rule": rule
                }
                context_event(stage="decision_end", decision=result)
                return result
            failure_reasons.append(f"rule#{idx}:{reason}")

        result = {
            "should_reply": False,
            "reason": "; ".join(failure_reasons) if failure_reasons else "规则未触发",
            "matched_rule": None,
        }
        context_event(stage="decision_end", decision=result)
        return result

    def evaluate_trigger_expression(
        self,
        expression: str,
        rule: dict[str, Any],
        context: DidaAgentMessageContext,
    ) -> tuple[bool, str]:
        allowed_conditions = {
            "at_bot": lambda: self.condition_at_bot(context),
            "keyword": lambda: self.condition_keyword(rule, context),
            "always": lambda: self.condition_always(),
            "ai_decide": lambda: self.condition_ai_decide(rule, context),
        }

        tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression))
        invalid_tokens = [token for token in tokens if token not in allowed_conditions]
        if invalid_tokens:
            invalid_text = ",".join(sorted(invalid_tokens))
            return False, "trigger_mode 包含未知条件: " + invalid_text

        condition_values: dict[str, bool] = {}
        condition_reasons: dict[str, str] = {}
        for token in tokens:
            value, reason = allowed_conditions[token]()
            condition_values[token] = bool(value)
            condition_reasons[token] = reason

        py_expr = expression.replace("&&", " and ").replace("||", " or ")
        py_expr = re.sub(r"!\s*", " not ", py_expr)

        try:
            result = bool(eval(py_expr, {"__builtins__": {}}, condition_values))
        except Exception as error:
            return False, f"trigger_mode 解析失败: {error}"

        if result:
            return True, f"命中表达式: {expression}"

        reason_text = ", ".join([f"{name}={condition_values[name]}" for name in sorted(condition_values)])
        detail_text = "; ".join([f"{name}:{condition_reasons[name]}" for name in sorted(condition_reasons)])
        return False, f"表达式未命中: {reason_text}; {detail_text}"

    def condition_always(self) -> tuple[bool, str]:
        return True, "always=true"

    def condition_at_bot(self, context: DidaAgentMessageContext) -> tuple[bool, str]:
        if context.chat_type != "group":
            return False, "私聊场景不满足 at_bot"

        at_ids = re.findall(r"\[CQ:at,qq=(\d+)\]", context.raw_message or "")
        if not at_ids:
            return False, "消息中没有 @"

        configured_bot_qq = str(self.config.get("bot_qq", "")).strip()
        if configured_bot_qq:
            if configured_bot_qq in at_ids:
                return True, f"命中 @bot({configured_bot_qq})"
            return False, f"@目标不含 bot_qq={configured_bot_qq}"

        return True, "检测到 @，且未配置 bot_qq，按命中处理"

    def condition_keyword(self, rule: dict[str, Any], context: DidaAgentMessageContext) -> tuple[bool, str]:
        keywords = rule.get("keywords", [])
        if not isinstance(keywords, list) or not keywords:
            return False, "keywords 为空"

        text = (context.cleaned_message or "")
        for keyword in keywords:
            keyword_text = str(keyword).strip()
            if keyword_text and keyword_text in text:
                return True, f"命中关键词: {keyword_text}"
        return False, "未命中关键词"

    def condition_ai_decide(self, rule: dict[str, Any], context: DidaAgentMessageContext) -> tuple[bool, str]:
        prompt = str(rule.get("ai_decision_prompt") or "").strip()
        if not prompt:
            return False, "缺少 ai_decision_prompt"

        model_name = str(rule.get("decision_model") or rule.get("model") or self.model)

        if load_dotenv is not None:
            load_dotenv(override=False)
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            return False, "缺少 LLM_API_KEY"

        rule_temp = rule.get("temperature")
        if rule_temp is not None:
            try:
                temperature = float(rule_temp)
            except (TypeError, ValueError):
                temperature = self.temperature
        else:
            temperature = self.temperature
        
        llm_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "temperature": temperature,
            "openai_api_key": api_key,
            "max_tokens": 1024,
        }
        base_url = os.getenv("LLM_API_BASE_URL")
        if base_url:
            llm_kwargs["openai_api_base"] = base_url

        llm = ChatOpenAI(**llm_kwargs).with_structured_output(DidaAgentAIDecision)

        def decide_node(state: DidaAgentAIState) -> DidaAgentAIState:
            llm_input = {
                "chat_type": state["chat_type"],
                "group_id": state["group_id"],
                "user_id": state["user_id"],
                "user_name": state["user_name"],
                "ts": state["ts"],
                "raw_message": state["raw_message"],
                "cleaned_message": state["cleaned_message"],
                "history_messages": state["history_messages"],
            }
            result = llm.invoke(
                [
                    SystemMessage(content=state["prompt"]),
                    HumanMessage(content=json.dumps(llm_input, ensure_ascii=False)),
                ]
            )
            return {
                **state,
                "should_reply": bool(result.should_reply),
                "reason": str(result.reason or "").strip(),
            }

        graph = StateGraph(DidaAgentAIState)
        graph.add_node("decide", decide_node)
        graph.add_edge(START, "decide")
        graph.add_edge("decide", END)
        app = graph.compile()
        context_event = bind_agent_event(
            agent_name="dida_agent",
            task_type="DIDA_AGENT",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
        )

        context_event(
            stage="ai_decide_start",
            extra={"model": self.model, "history_count": len(context.history_messages)},
        )

        final_state = app.invoke(
            {
                "prompt": prompt,
                "chat_type": context.chat_type,
                "group_id": context.group_id,
                "user_id": context.user_id,
                "user_name": context.user_name,
                "ts": context.ts,
                "raw_message": context.raw_message,
                "cleaned_message": context.cleaned_message,
                "history_messages": context.history_messages,
                "should_reply": False,
                "reason": "",
            }
        )

        should_reply = bool(final_state.get("should_reply", False))
        reason = str(final_state.get("reason", "")).strip()
        context_event(stage="ai_decide_end", decision={"should_reply": should_reply, "reason": reason})
        if should_reply:
            return True, reason or "ai_decide=true"
        return False, reason or "ai_decide=false"

    def _load_task_context_data(self, context: DidaAgentMessageContext) -> list[dict[str, Any]]:
        """从 dida_context.json 加载任务数据（结构化）。"""
        tasks_data: list[dict[str, Any]] = []
        try:
            context_path = os.path.join("data", "dida_context.json")
            if not os.path.exists(context_path):
                return []
            
            with open(context_path, "r", encoding="utf-8") as f:
                dida_data = json.load(f)
            
            if not isinstance(dida_data, dict):
                return []

            target_uids = {str(context.user_id)}
            
            # Admin permission check
            runtime_config = get_dida_agent_runtime_config()
            admin_qqs = runtime_config.get("admin_qqs", [])
            is_admin = str(context.user_id) in admin_qqs
            
            if is_admin:
                at_matches = re.findall(r"\[CQ:at,qq=(\d+)\]", context.raw_message or "")
                for uid in at_matches:
                    target_uids.add(str(uid))

            for uid, tasks in dida_data.items():
                if str(uid) not in target_uids:
                    continue
                if not isinstance(tasks, list) or not tasks:
                    continue
                tasks_data.extend(tasks)
        except Exception:
            pass
        return tasks_data

    def generate_action(
        self,
        *,
        action_prompt: str,
        context: DidaAgentMessageContext,
        rule: dict[str, Any] | None = None,
    ) -> ActionLLMResult:
        """Model A: 生成结构化动作（或澄清请求）。"""
        prompt = str(action_prompt).strip()
        tasks_list = self._load_task_context_data(context)
        
        # Build structured input for Model A
        llm_input = {
            "meta": {
                "chat_type": context.chat_type,
                "group_id": context.group_id,
                "user_id": context.user_id,
                "user_name": context.user_name,
                "ts": context.ts,
            },
            "instruction": {
                "raw_message": context.raw_message,
                "cleaned_message": context.cleaned_message,
            },
            "tasks": tasks_list
        }

        # Model selection
        model_name = str(rule.get("action_model") or rule.get("model") or self.model) if rule else self.model
        
        if load_dotenv is not None:
            load_dotenv(override=False)
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("缺少 LLM_API_KEY")

        temperature = self.temperature
        if rule is not None:
            try:
                # 优先使用 action_temperature，其次 temperature
                rule_temp = rule.get("action_temperature")
                if rule_temp is None:
                    rule_temp = rule.get("temperature")
                if rule_temp is not None:
                    temperature = float(rule_temp)
            except (TypeError, ValueError):
                pass

        llm_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "temperature": temperature,
            "openai_api_key": api_key,
            "max_tokens": 1024,
        }
        base_url = os.getenv("LLM_API_BASE_URL")
        if base_url:
            llm_kwargs["openai_api_base"] = base_url

        llm_base = ChatOpenAI(**llm_kwargs)
        try:
            llm = llm_base.with_structured_output(ActionLLMResult)
        except TypeError:
            # Fallback if structured output not supported
            llm = llm_base

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(llm_input, ensure_ascii=False)),
        ]
        
        print(f"[DIDA_DEBUG] generate_action: invoking LLM. model={model_name} run_id={context.run_id}")
        try:
            print(f"[DIDA_DEBUG] generate_action input (preview): {json.dumps(llm_input, ensure_ascii=False)[:1000]}")
        except Exception:
            pass

        try:
            t0 = perf_counter()
            result = llm.invoke(messages)
            t1 = perf_counter()
            print(f"[DIDA_DEBUG] generate_action: LLM returned. cost={t1-t0:.3f}s run_id={context.run_id}")
            
            try:
                if hasattr(result, "model_dump_json"):
                    print(f"[DIDA_DEBUG] generate_action output: {result.model_dump_json(ensure_ascii=False)}")
                else:
                    print(f"[DIDA_DEBUG] generate_action output (raw): {result}")
            except Exception as log_err:
                print(f"[DIDA_DEBUG] generate_action output log failed: {log_err}")

            if isinstance(result, ActionLLMResult):
                return result
            # Handle raw AIMessage fallback if structured output failed silently
            return ActionLLMResult(dida_action=None, need_clarification=True, clarification_question="系统内部错误：模型输出解析失败")
        except Exception as e:
            print(f"[DIDA_AGENT-ACTION-ERROR] {e}")
            return ActionLLMResult(dida_action=None, need_clarification=True, clarification_question=f"系统错误：{str(e)}")

    def generate_final_reply(
        self,
        *,
        reply_prompt: str,
        context: DidaAgentMessageContext,
        action_result: ActionLLMResult | None,
        execution_result: dict[str, Any] | None,
        rule: dict[str, Any] | None = None,
    ) -> ReplyResponse:
        """Model B: 根据执行结果生成最终回复。"""
        prompt = str(reply_prompt).strip()
        
        # Build structured input for Model B
        action_summary = {}
        if action_result and action_result.dida_action:
            act = action_result.dida_action
            action_summary = {
                "action_type": act.action_type,
                "title": act.title,
                "project_id": act.project_id,
                "due_date": act.due_date,
                "is_all_day": act.is_all_day
            }
        
        llm_input = {
            "meta": {
                "chat_type": context.chat_type,
                "group_id": context.group_id,
                "user_id": context.user_id,
                "user_name": context.user_name,
                "ts": context.ts,
            },
            "instruction": {
                "raw_message": context.raw_message,
                "cleaned_message": context.cleaned_message,
            },
            "action": action_summary,
            "execution": execution_result or {"ok": False, "message": "未执行", "data": None}
        }
        
        if action_result and action_result.need_clarification:
            llm_input["execution"] = {
                "ok": False,
                "message": f"需要澄清：{action_result.clarification_question}",
                "need_clarification": True
            }

        # Model selection
        model_name = str(rule.get("reply_model") or rule.get("model") or self.model) if rule else self.model
        
        if load_dotenv is not None:
            load_dotenv(override=False)
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("缺少 LLM_API_KEY")

        temperature = self.temperature
        if rule is not None:
            try:
                rule_temp = rule.get("reply_temperature")
                if rule_temp is None:
                    rule_temp = rule.get("temperature")
                if rule_temp is not None:
                    temperature = float(rule_temp)
            except (TypeError, ValueError):
                pass

        llm_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "temperature": temperature,
            "openai_api_key": api_key,
            "max_tokens": 1024,
        }
        base_url = os.getenv("LLM_API_BASE_URL")
        if base_url:
            llm_kwargs["openai_api_base"] = base_url

        llm_base = ChatOpenAI(**llm_kwargs)
        # Model B should output simple text usually, but we defined ReplyResponse
        try:
            llm = llm_base.with_structured_output(ReplyResponse)
        except TypeError:
            llm = llm_base

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(llm_input, ensure_ascii=False)),
        ]
        
        print(f"[DIDA_DEBUG] generate_final_reply: invoking LLM. model={model_name} run_id={context.run_id}")
        try:
            print(f"[DIDA_DEBUG] generate_final_reply input (preview): {json.dumps(llm_input, ensure_ascii=False)[:1000]}")
        except Exception:
            pass
        
        try:
            t0 = perf_counter()
            result = llm.invoke(messages)
            t1 = perf_counter()
            print(f"[DIDA_DEBUG] generate_final_reply: LLM returned. cost={t1-t0:.3f}s run_id={context.run_id}")
            
            try:
                if hasattr(result, "model_dump_json"):
                    print(f"[DIDA_DEBUG] generate_final_reply output: {result.model_dump_json(ensure_ascii=False)}")
                else:
                    print(f"[DIDA_DEBUG] generate_final_reply output (raw): {result}")
            except Exception as log_err:
                print(f"[DIDA_DEBUG] generate_final_reply output log failed: {log_err}")

            if isinstance(result, ReplyResponse):
                return result
            # Fallback
            if hasattr(result, "content"):
                return ReplyResponse(reply_text=str(result.content))
            return ReplyResponse(reply_text="（系统错误：无法生成回复）")
        except Exception as e:
            print(f"[DIDA_AGENT-REPLY-ERROR] {e}")
            return ReplyResponse(reply_text=f"（系统错误：{e}）")

    def generate_reply_text(
        self,
        *,
        reply_prompt: str,
        context: DidaAgentMessageContext,
        rule: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = str(reply_prompt).strip()
        
        # Inject Dida task context if available
        try:
            context_path = os.path.join("data", "dida_context.json")
            if os.path.exists(context_path):
                with open(context_path, "r", encoding="utf-8") as f:
                    dida_data = json.load(f)
                    
                task_context_lines = []
                # Check for user-specific context or fallback to all
                # The keys in dida_context.json are internal user IDs (e.g., from dida_tokens.json)
                # Since dida_agent doesn't know the internal ID mapping easily without auth,
                # we will include all available tasks from the context file.
                # In a single-user/couple bot scenario, this is usually fine.
                
                # Only include tasks for the current user to avoid context pollution and privacy issues
                target_uids = {str(context.user_id)}
                
                # Admin permission check
                runtime_config = get_dida_agent_runtime_config()
                admin_qqs = runtime_config.get("admin_qqs", [])
                is_admin = str(context.user_id) in admin_qqs
                
                if is_admin:
                    # If admin, allow seeing tasks of mentioned users
                    at_matches = re.findall(r"\[CQ:at,qq=(\d+)\]", context.raw_message or "")
                    for uid in at_matches:
                        target_uids.add(str(uid))

                for uid, tasks in dida_data.items():
                    if str(uid) not in target_uids:
                        continue
                    if not isinstance(tasks, list) or not tasks:
                        continue
                    for t in tasks:
                        tid = str(t.get("id") or "")
                        ttitle = str(t.get("title") or "")
                        tproject = str(t.get("project") or "")
                        tdue_raw = str(t.get("due") or "")
                        tdue = "无到期"
                        if tdue_raw:
                            try:
                                # Clean up potential "Z" for isoformat
                                _dt = datetime.fromisoformat(tdue_raw.replace("Z", "+00:00"))
                                # Convert to local
                                dt_local = _dt.astimezone()
                                if bool(t.get("isAllDay")):
                                    tdue = dt_local.strftime("%Y-%m-%d") + " (全天)"
                                else:
                                    tdue = dt_local.strftime("%Y-%m-%d %H:%M")
                            except ValueError:
                                tdue = tdue_raw
                        # Format: [Project] Title (ID: ...) Due: ...
                        task_context_lines.append(f"- [{tproject}] {ttitle} (ID: {tid}) Due: {tdue}")
                
                if task_context_lines:
                    prompt += "\n\n【当前 Dida 任务列表（仅供 AI 参考，不要泄露 ID 给用户，但在调用工具时必须使用 ID）】:\n" + "\n".join(task_context_lines)
                
                if is_admin:
                    prompt += "\n\n【权限提醒】你是管理员，可以操作他人的任务。若指令涉及他人（如“查看@A的任务”），请在 DidaAction 中准确填写 target_user_id（即被 @ 的用户 QQ 号）。"
        except Exception as e:
            pass # Ignore context loading errors

        if not prompt:
            return {"reply_text": "", "dida_action": None}

        model_name = str(rule.get("reply_model") or rule.get("model") or self.model) if rule else self.model

        if load_dotenv is not None:
            load_dotenv(override=False)
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("缺少 LLM_API_KEY")

        temperature = self.temperature
        
        if rule is not None:
            rule_temp = rule.get("temperature")
            if rule_temp is not None:
                try:
                    temperature = float(rule_temp)
                except (TypeError, ValueError):
                    pass  # 转换失败则保持全局值

        llm_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "temperature": temperature,
            "openai_api_key": api_key,
        }
        base_url = os.getenv("LLM_API_BASE_URL")
        if base_url:
            llm_kwargs["openai_api_base"] = base_url

        llm_base = ChatOpenAI(**llm_kwargs)
        use_raw_fallback = True
        try:
            llm = llm_base.with_structured_output(DidaAgentGeneratedReply, include_raw=True)
        except TypeError:
            use_raw_fallback = False
            llm = llm_base.with_structured_output(DidaAgentGeneratedReply)

        def generate_node(state: DidaAgentGenerateState) -> DidaAgentGenerateState:
            llm_input = {
                "chat_type": state["chat_type"],
                "group_id": state["group_id"],
                "user_id": state["user_id"],
                "user_name": state["user_name"],
                "ts": state["ts"],
                "raw_message": state["raw_message"],
                "cleaned_message": state["cleaned_message"],
                "history_messages": state["history_messages"],
            }
            messages = [
                SystemMessage(content=state["prompt"]),
                HumanMessage(content=json.dumps(llm_input, ensure_ascii=False)),
            ]
            try:
                result = llm.invoke(messages)
            except Exception:
                # 某些兼容模型不会遵守结构化输出，改走原始文本兜底，避免 reply_len=0
                fallback_reply = ""
                try:
                    raw_result = llm_base.invoke(messages)
                    fallback_reply = _extract_reply_text_from_raw_output(raw_result)
                except Exception:
                    fallback_reply = ""
                return {
                    **state,
                    "reply_text": fallback_reply or _default_reply_when_parse_failed(),
                    "dida_action": None,
                }
            if use_raw_fallback and isinstance(result, dict):
                parsed = result.get("parsed")
                if parsed is not None:
                    return {
                        **state,
                        "reply_text": str(getattr(parsed, "reply_text", "") or "").strip(),
                        "dida_action": getattr(parsed, "dida_action", None),
                    }
                raw_output = result.get("raw")
                fallback_reply = _extract_reply_text_from_raw_output(raw_output)
                if fallback_reply:
                    return {
                        **state,
                        "reply_text": fallback_reply,
                        "dida_action": None,
                    }
                parse_error = result.get("parsing_error")
                if parse_error:
                    return {
                        **state,
                        "reply_text": _default_reply_when_parse_failed(),
                        "dida_action": None,
                    }
                return {
                    **state,
                    "reply_text": _default_reply_when_parse_failed(),
                    "dida_action": None,
                }
            structured_reply = str(getattr(result, "reply_text", "") or "").strip()
            if structured_reply:
                return {
                    **state,
                    "reply_text": structured_reply,
                    "dida_action": getattr(result, "dida_action", None),
                }
            fallback_reply = _extract_reply_text_from_raw_output(result)
            return {
                **state,
                "reply_text": fallback_reply or _default_reply_when_parse_failed(),
                "dida_action": None,
            }

        graph = StateGraph(DidaAgentGenerateState)
        graph.add_node("generate", generate_node)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", END)
        app = graph.compile()
        context_event = bind_agent_event(
            agent_name="dida_agent",
            task_type="DIDA_AGENT",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
        )

        context_event(
            stage="reply_generate_start",
            extra={"model": model_name, "history_count": len(context.history_messages)},
        )

        final_state = app.invoke(
            {
                "prompt": prompt,
                "chat_type": context.chat_type,
                "group_id": context.group_id,
                "user_id": context.user_id,
                "user_name": context.user_name,
                "ts": context.ts,
                "raw_message": context.raw_message,
                "cleaned_message": context.cleaned_message,
                "history_messages": context.history_messages,
                "reply_text": "",
                "dida_action": None,
            }
        )

        reply_text = str(final_state.get("reply_text", "")).strip()
        dida_action = final_state.get("dida_action")
        context_event(stage="reply_generate_end", decision={"reply_length": len(reply_text), "has_reply": bool(reply_text)})
        return {"reply_text": reply_text, "dida_action": dida_action}


def run_dida_agent_pipeline(
    *,
    chat_type: str,
    group_id: str,
    user_id: str,
    user_name: str,
    ts: str,
    raw_message: str,
    cleaned_message: str,
    run_id: str = "",
) -> dict[str, Any]:
    """DidaAgent 主处理管道：先判定，再按规则提示词生成回复文本。"""
    runtime_config = get_dida_agent_runtime_config()
    context_limit = runtime_config["context_history_limit"]
    context_max_chars = runtime_config["context_max_chars"]
    context_window_seconds = runtime_config["context_window_seconds"]

    context_messages = load_recent_context_messages(
        chat_type=str(chat_type).strip().lower(),
        group_id=str(group_id),
        user_id=str(user_id),
        current_ts=str(ts),
        current_cleaned_message=str(cleaned_message),
        limit=max(context_limit, 0),
        max_chars=max(context_max_chars, 0),
        window_seconds=max(context_window_seconds, 0),
    )

    context = DidaAgentMessageContext(
        chat_type=str(chat_type).strip().lower(),
        group_id=str(group_id),
        user_id=str(user_id),
        user_name=str(user_name),
        ts=str(ts),
        raw_message=str(raw_message),
        cleaned_message=str(cleaned_message),
        history_messages=context_messages,
        run_id=str(run_id),
    )
    context_event = bind_agent_event(
        agent_name="dida_agent",
        task_type="DIDA_AGENT",
        run_id=context.run_id,
        chat_type=context.chat_type,
        group_id=context.group_id,
        user_id=context.user_id,
        user_name=context.user_name,
        ts=context.ts,
    )

    context_event(
        stage="context_loaded",
        extra={
            "history_count": len(context_messages),
            "context_max_chars": context_max_chars,
            "context_window_seconds": context_window_seconds,
        },
    )

    # Performance checkpoint
    t0 = perf_counter()
    print(f"[DIDA_DEBUG] run_dida_agent_pipeline start check. run_id={run_id}")

    # 先判断是否要回复
    engine = DidaAgentDecisionEngine(DIDA_AGENT_CONFIG)
    result = engine.should_reply(context)
    
    t1 = perf_counter()
    print(f"[DIDA_DEBUG] should_reply finished. cost={t1-t0:.3f}s result={bool(result.get('should_reply'))} run_id={run_id}")

    # 然后生成具体的回复内容文本
    reply_payload: dict[str, Any] = {"reply_text": "", "dida_action": None}
    
    rule = result.get("rule") if isinstance(result.get("rule"), dict) else {}
    action_prompt = str(rule.get("action_prompt", "")).strip()
    is_new_mode = bool(action_prompt) and bool(rule.get("dida_enabled", False))

    if bool(result.get("should_reply", False)):
        if is_new_mode:
             # Model A (Action LLM)
             try:
                 print(f"[DIDA_DEBUG] generate_action start. run_id={run_id}")
                 action_res = engine.generate_action(action_prompt=action_prompt, context=context, rule=rule)
                 t2 = perf_counter()
                 print(f"[DIDA_DEBUG] generate_action finished. cost={t2-t1:.3f}s run_id={run_id}")
                 
                 return {
                     "should_reply": True,
                     "reason": str(result.get("reason", "")),
                     "matched_rule": result.get("matched_rule"),
                     "trigger_mode": result.get("trigger_mode", ""),
                     "is_new_mode": True,
                     "action_result": action_res,
                     "context": context,
                     "rule": rule,
                     "engine": engine, 
                     "chat_type": context.chat_type,
                     "group_id": context.group_id,
                     "user_id": context.user_id,
                 }
             except Exception as error:
                 context_event(stage="action_generate_error", error=str(error))
                 print(f"[DIDA_AGENT-ERROR] generate_action failed: {error}")
                 import traceback
                 traceback.print_exc()
                 return {
                     "should_reply": False, 
                     "reason": f"action_error: {error}",
                     "chat_type": context.chat_type,
                     "group_id": context.group_id,
                     "user_id": context.user_id,
                 }

        reply_prompt = str(result.get("reply_prompt", "")).strip()
        if reply_prompt:
            try:
                reply_payload = engine.generate_reply_text(reply_prompt=reply_prompt, context=context, rule=rule)
            except Exception as error:
                context_event(stage="reply_generate_error", error=str(error))
    dida_action = reply_payload.get("dida_action")
    if not bool(rule.get("dida_enabled", False)):
        dida_action = None
        reply_payload["dida_action"] = None
    return {
        "should_reply": bool(result.get("should_reply", False)),
        "reason": str(result.get("reason", "")),
        "matched_rule": result.get("matched_rule"),
        "trigger_mode": result.get("trigger_mode", ""),
        "reply_text": str(reply_payload.get("reply_text", "") or ""),
        "dida_action": dida_action,
        "chat_type": context.chat_type,
        "group_id": context.group_id,
        "user_id": context.user_id,
        "is_new_mode": False
    }
