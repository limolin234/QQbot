"""AutoReply 工作流：当前只实现“是否需要回复”的判定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypedDict
import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from workflows.agent_observe import observe_agent_event
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

        observe_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
            stage="rules_filtered",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
            extra={"configured_rules": len(self.rules), "matched_rules": len(matching_rules)},
        )

        if not matching_rules:
            result = {
                "should_reply": False,
                "reason": "未命中启用规则",
                "matched_rule": None,
            }
            observe_agent_event(
                agent_name="auto_reply",
                task_type="AUTO_REPLY",
                stage="decision_end",
                run_id=context.run_id,
                chat_type=context.chat_type,
                group_id=context.group_id,
                user_id=context.user_id,
                user_name=context.user_name,
                ts=context.ts,
                decision=result,
            )
            return result

        failure_reasons: list[str] = []
        for idx, rule in enumerate(matching_rules):
            expression = str(rule.get("trigger_mode") or "always").strip()
            ok, reason = self.evaluate_trigger_expression(expression, rule, context)
            observe_agent_event(
                agent_name="auto_reply",
                task_type="AUTO_REPLY",
                stage="rule_evaluated",
                run_id=context.run_id,
                chat_type=context.chat_type,
                group_id=context.group_id,
                user_id=context.user_id,
                user_name=context.user_name,
                ts=context.ts,
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
                }
                observe_agent_event(
                    agent_name="auto_reply",
                    task_type="AUTO_REPLY",
                    stage="decision_end",
                    run_id=context.run_id,
                    chat_type=context.chat_type,
                    group_id=context.group_id,
                    user_id=context.user_id,
                    user_name=context.user_name,
                    ts=context.ts,
                    decision=result,
                )
                return result
            failure_reasons.append(f"rule#{idx}:{reason}")

        result = {
            "should_reply": False,
            "reason": "; ".join(failure_reasons) if failure_reasons else "规则未触发",
            "matched_rule": None,
        }
        observe_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
            stage="decision_end",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
            decision=result,
        )
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

        if load_dotenv is not None:
            load_dotenv(override=False)
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            return False, "缺少 LLM_API_KEY"

        llm_kwargs: dict[str, Any] = {
            "model_name": self.model,
            "temperature": self.temperature,
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

        observe_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
            stage="ai_decide_start",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
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
        observe_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
            stage="ai_decide_end",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
            decision={"should_reply": should_reply, "reason": reason},
        )
        if should_reply:
            return True, reason or "ai_decide=true"
        return False, reason or "ai_decide=false"

    def generate_reply_text(
        self,
        *,
        reply_prompt: str,
        context: AutoReplyMessageContext,
    ) -> str:
        prompt = str(reply_prompt).strip()
        if not prompt:
            return ""

        if load_dotenv is not None:
            load_dotenv(override=False)
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("缺少 LLM_API_KEY")

        llm_kwargs: dict[str, Any] = {
            "model_name": self.model,
            "temperature": self.temperature,
            "openai_api_key": api_key,
        }
        base_url = os.getenv("LLM_API_BASE_URL")
        if base_url:
            llm_kwargs["openai_api_base"] = base_url

        llm = ChatOpenAI(**llm_kwargs).with_structured_output(AutoReplyGeneratedReply)

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
            result = llm.invoke(
                [
                    SystemMessage(content=state["prompt"]),
                    HumanMessage(content=json.dumps(llm_input, ensure_ascii=False)),
                ]
            )
            return {
                **state,
                "reply_text": str(result.reply_text or "").strip(),
            }

        graph = StateGraph(AutoReplyGenerateState)
        graph.add_node("generate", generate_node)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", END)
        app = graph.compile()

        observe_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
            stage="reply_generate_start",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
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
                "reply_text": "",
            }
        )

        reply_text = str(final_state.get("reply_text", "")).strip()
        observe_agent_event(
            agent_name="auto_reply",
            task_type="AUTO_REPLY",
            stage="reply_generate_end",
            run_id=context.run_id,
            chat_type=context.chat_type,
            group_id=context.group_id,
            user_id=context.user_id,
            user_name=context.user_name,
            ts=context.ts,
            decision={"reply_length": len(reply_text), "has_reply": bool(reply_text)},
        )
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

    observe_agent_event(
        agent_name="auto_reply",
        task_type="AUTO_REPLY",
        stage="context_loaded",
        run_id=context.run_id,
        chat_type=context.chat_type,
        group_id=context.group_id,
        user_id=context.user_id,
        user_name=context.user_name,
        ts=context.ts,
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
            try:
                reply_text = engine.generate_reply_text(reply_prompt=reply_prompt, context=context)
            except Exception as error:
                observe_agent_event(
                    agent_name="auto_reply",
                    task_type="AUTO_REPLY",
                    stage="reply_generate_error",
                    run_id=context.run_id,
                    chat_type=context.chat_type,
                    group_id=context.group_id,
                    user_id=context.user_id,
                    user_name=context.user_name,
                    ts=context.ts,
                    error=str(error),
                )
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
