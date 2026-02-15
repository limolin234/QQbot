"""AutoReply 工作流：当前只实现“是否需要回复”的判定。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, TypedDict
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
from workflows.agent_observe import bind_agent_event, generate_run_id
from workflows.agent_config_loader import load_current_agent_config

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


AUTO_REPLY_CONFIG = load_current_agent_config(__file__)


def get_auto_reply_runtime_config() -> dict[str, Any]:
    """读取并规范化 auto_reply 运行时配置。"""

    def int_config(name: str, default: int) -> int:
        try:
            value = int(AUTO_REPLY_CONFIG.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(value, 0)

    def bool_config(name: str, default: bool) -> bool:
        value = AUTO_REPLY_CONFIG.get(name, default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    return {
        "context_history_limit": int_config("context_history_limit", 12),
        "context_max_chars": int_config("context_max_chars", 1800),
        "context_window_seconds": int_config("context_window_seconds", 1800),
        "min_reply_interval_seconds": int_config("min_reply_interval_seconds", 300),
        "flush_check_interval_seconds": int_config("flush_check_interval_seconds", 10),
        "pending_expire_seconds": int_config("pending_expire_seconds", 3600),
        "pending_max_messages": int_config("pending_max_messages", 50),
        "bypass_cooldown_when_at_bot": bool_config("bypass_cooldown_when_at_bot", True),
        "bot_qq": str(AUTO_REPLY_CONFIG.get("bot_qq", "")).strip(),
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


def get_auto_reply_monitor_numbers(chat_type: str = "group") -> set[str]:
    """读取 auto_reply 配置里启用规则的监控号码集合。"""
    rules = AUTO_REPLY_CONFIG.get("rules", [])
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


class AutoReplyDispatcher:
    """AutoReply 接入层：监控命中 + 冷却窗口 + pending 聚合/过期。"""

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
        monitor_numbers = get_auto_reply_monitor_numbers(chat_type=target_chat_type)
        if normalized_monitor_value not in monitor_numbers:
            return False

        runtime_config = get_auto_reply_runtime_config()
        min_reply_interval = max(int(runtime_config.get("min_reply_interval_seconds", 300)), 0)
        pending_max_messages = max(int(runtime_config.get("pending_max_messages", 50)), 1)

        auto_reply_payload = self._build_payload(
            raw_message=str(raw_message),
            cleaned_message=str(cleaned_message),
            chat_type=target_chat_type,
            group_id=normalized_monitor_value if target_chat_type == "group" else "private",
            user_id=str(user_id),
            user_name=str(user_name),
            ts=str(ts),
        )
        log_event = bind_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
            chat_type=target_chat_type,
            group_id=auto_reply_payload["group_id"],
            user_id=auto_reply_payload["user_id"],
            user_name=auto_reply_payload["user_name"],
            ts=auto_reply_payload["ts"],
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
                    state["pending_payload"] = auto_reply_payload
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
            await enqueue_payload(auto_reply_payload)
        return True

    async def pending_worker(self, *, enqueue_payload: Callable[[dict[str, str]], Awaitable[None]]) -> None:
        while True:
            runtime_config = get_auto_reply_runtime_config()
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
                        agent_name="auto_reply",
                        task_type="AUTO_REPLY",
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


_AUTO_REPLY_DISPATCHER = AutoReplyDispatcher()


def get_auto_reply_dispatcher() -> AutoReplyDispatcher:
    return _AUTO_REPLY_DISPATCHER


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


async def _execute_auto_reply_payload(payload: dict[str, str]) -> None:
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
        agent_name="auto_reply",
        task_type="AUTO_REPLY",
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
            run_auto_reply_pipeline,
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
        reply_text = str(result.get("reply_text", "") or "").strip()

        if should_reply_flag and reply_text:
            log_event(stage="send_start", extra={"reply_length": len(reply_text)})
            try:
                if chat_type == "group":
                    await bot.api.post_group_msg(group_id, text=reply_text)
                else:
                    await bot.api.post_private_msg(user_id, text=reply_text)
                log_event(stage="send_end", decision={"sent": True, "reply_length": len(reply_text)})
            except Exception as send_error:
                log_event(stage="send_error", error=str(send_error))
                print(
                    "[AUTO_REPLY-SEND-ERROR] "
                    f"chat={chat_type} group={group_id} user={user_id} error={send_error}"
                )
        elif should_reply_flag:
            log_event(stage="send_skip", decision={"sent": False, "reason": "reply_text_empty"})

        print(
            "[AUTO_REPLY] "
            f"chat={chat_type} group={group_id} user={user_name}({user_id}) "
            f"should_reply={should_reply_flag} "
            f"reason={reason_text} reply_len={len(reply_text)}"
        )
    except Exception as error:
        elapsed_ms = (perf_counter() - started) * 1000
        log_event(stage="error", latency_ms=elapsed_ms, error=str(error))
        print(
            "[AUTO_REPLY-ERROR] "
            f"chat={chat_type} group={group_id} user={user_id} error={error}"
        )


def _create_auto_reply_task(payload: dict[str, str]) -> None:
    task = asyncio.create_task(_execute_auto_reply_payload(payload))

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except Exception as error:
            print(f"[AUTO_REPLY-ASYNC-ERROR] error={error}")

    task.add_done_callback(_on_done)


async def _enqueue_auto_reply_payload(payload: dict[str, str]) -> None:
    _create_auto_reply_task(payload)


async def auto_reply_pending_worker() -> None:
    dispatcher = get_auto_reply_dispatcher()
    await dispatcher.pending_worker(enqueue_payload=_enqueue_auto_reply_payload)


async def enqueue_auto_reply_if_monitored(
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
    dispatcher = get_auto_reply_dispatcher()
    return await dispatcher.enqueue_if_monitored(
        chat_type=target_chat_type,
        monitor_value=monitor_value,
        raw_message=raw_message,
        cleaned_message=clean_message(raw_message),
        user_id=str(getattr(msg, "user_id", "unknown")),
        user_name=_extract_user_name(msg),
        ts=_extract_message_ts(msg),
        enqueue_payload=_enqueue_auto_reply_payload,
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
class AutoReplyMessageContext:
    chat_type: str
    group_id: str
    user_id: str
    user_name: str
    ts: str
    raw_message: str
    cleaned_message: str
    history_messages: list[str] = field(default_factory=list)
    run_id: str = ""


class AutoReplyAIDecision(BaseModel):
    should_reply: bool = Field(description="是否需要回复")
    reason: str = Field(default="", description="判定理由")


class AutoReplyGeneratedReply(BaseModel):
    reply_text: str = Field(default="", description="自动回复文本")


class AutoReplyAIState(TypedDict):
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


class AutoReplyGenerateState(TypedDict):
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


class AutoReplyDecisionEngine:
    """自动回复判定器：负责读取规则表达式并计算 should_reply。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config if isinstance(config, dict) else {}
        raw_rules = self.config.get("rules", [])
        self.rules = [rule for rule in raw_rules if isinstance(rule, dict) and bool(rule.get("enabled", False))]
        self.model = str(self.config.get("model") or "gpt-4o-mini")
        try:
            self.temperature = float(self.config.get("temperature", 0.0))
        except (TypeError, ValueError):
            self.temperature = 0.0

    def should_reply(self, context: AutoReplyMessageContext) -> dict[str, Any]:
        context_event = bind_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
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
        context: AutoReplyMessageContext,
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

    def condition_at_bot(self, context: AutoReplyMessageContext) -> tuple[bool, str]:
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

    def condition_keyword(self, rule: dict[str, Any], context: AutoReplyMessageContext) -> tuple[bool, str]:
        keywords = rule.get("keywords", [])
        if not isinstance(keywords, list) or not keywords:
            return False, "keywords 为空"

        text = (context.cleaned_message or "")
        for keyword in keywords:
            keyword_text = str(keyword).strip()
            if keyword_text and keyword_text in text:
                return True, f"命中关键词: {keyword_text}"
        return False, "未命中关键词"

    def condition_ai_decide(self, rule: dict[str, Any], context: AutoReplyMessageContext) -> tuple[bool, str]:
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
        }
        base_url = os.getenv("LLM_API_BASE_URL")
        if base_url:
            llm_kwargs["openai_api_base"] = base_url

        llm = ChatOpenAI(**llm_kwargs).with_structured_output(AutoReplyAIDecision)

        def decide_node(state: AutoReplyAIState) -> AutoReplyAIState:
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

        graph = StateGraph(AutoReplyAIState)
        graph.add_node("decide", decide_node)
        graph.add_edge(START, "decide")
        graph.add_edge("decide", END)
        app = graph.compile()
        context_event = bind_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
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

    def generate_reply_text(
        self,
        *,
        reply_prompt: str,
        context: AutoReplyMessageContext,
        rule: dict[str, Any] | None = None,
    ) -> str:
        prompt = str(reply_prompt).strip()
        if not prompt:
            return ""

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
            llm = llm_base.with_structured_output(AutoReplyGeneratedReply, include_raw=True)
        except TypeError:
            use_raw_fallback = False
            llm = llm_base.with_structured_output(AutoReplyGeneratedReply)

        def generate_node(state: AutoReplyGenerateState) -> AutoReplyGenerateState:
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
                }
            if use_raw_fallback and isinstance(result, dict):
                parsed = result.get("parsed")
                if parsed is not None:
                    return {
                        **state,
                        "reply_text": str(getattr(parsed, "reply_text", "") or "").strip(),
                    }
                raw_output = result.get("raw")
                fallback_reply = _extract_reply_text_from_raw_output(raw_output)
                if fallback_reply:
                    return {
                        **state,
                        "reply_text": fallback_reply,
                    }
                parse_error = result.get("parsing_error")
                if parse_error:
                    return {
                        **state,
                        "reply_text": _default_reply_when_parse_failed(),
                    }
                return {
                    **state,
                    "reply_text": _default_reply_when_parse_failed(),
                }
            structured_reply = str(getattr(result, "reply_text", "") or "").strip()
            if structured_reply:
                return {
                    **state,
                    "reply_text": structured_reply,
                }
            fallback_reply = _extract_reply_text_from_raw_output(result)
            return {
                **state,
                "reply_text": fallback_reply or _default_reply_when_parse_failed(),
            }

        graph = StateGraph(AutoReplyGenerateState)
        graph.add_node("generate", generate_node)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", END)
        app = graph.compile()
        context_event = bind_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
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
            }
        )

        reply_text = str(final_state.get("reply_text", "")).strip()
        context_event(stage="reply_generate_end", decision={"reply_length": len(reply_text), "has_reply": bool(reply_text)})
        return reply_text


def run_auto_reply_pipeline(
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
    """AutoReply 主处理管道：先判定，再按规则提示词生成回复文本。"""
    runtime_config = get_auto_reply_runtime_config()
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

    context = AutoReplyMessageContext(
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
        agent_name="auto_reply",
        task_type="AUTO_REPLY",
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

    # 先判断是否要回复
    engine = AutoReplyDecisionEngine(AUTO_REPLY_CONFIG)
    result = engine.should_reply(context)

    # 然后生成具体的回复内容文本
    reply_text = ""
    if bool(result.get("should_reply", False)):
        reply_prompt = str(result.get("reply_prompt", "")).strip()
        if reply_prompt:
            rule = result.get("rule")
            try:
                reply_text = engine.generate_reply_text(reply_prompt=reply_prompt, context=context, rule=rule)
            except Exception as error:
                context_event(stage="reply_generate_error", error=str(error))
    return {
        "should_reply": bool(result.get("should_reply", False)),
        "reason": str(result.get("reason", "")),
        "matched_rule": result.get("matched_rule"),
        "trigger_mode": result.get("trigger_mode", ""),
        "reply_text": reply_text,
        "chat_type": context.chat_type,
        "group_id": context.group_id,
        "user_id": context.user_id,
    }
