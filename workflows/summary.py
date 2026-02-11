"""
Summary 工作流中的预处理与 LangGraph 汇总模块。

职责：
1) 解析 scheduler 生成的 `[group:...][user:...][name:...]` 来源块。
2) 清洗行文本（去空、截断、去重、限制最大行数）。
3) 通过最少节点的 LangGraph 执行 summary（Map -> Finalize）。

数据结构关系：
- SummaryBlock: 单个来源块（group/user + 行列表），是来源语义最小单元。
- SummaryChunk: 预处理全量结果，包含 blocks、可喂模型文本与统计信息。
- SummaryWorkflowPayload: agent 输入协议，复用 chunk 的内容并整理 stats。
- SummaryMapResult: 单个 chunk 的结构化摘要结果（LLM 输出映射）。
- SummaryFinalResult: 最终汇总结果（可直接格式化发送给用户）。

处理链路：
raw_message
  -> preprocess_summary_chunk
    -> _parse_blocks
    -> _collect_normalized_lines
  -> prepare_summary_payload
    -> SummaryWorkflowPayload
  -> run_summary_graph
    -> map_node (LLM with structured output)
    -> finalize_node (组装最终结果)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, TypedDict
import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .agent_config_loader import load_current_agent_config

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


UNKNOWN_GROUP = "unknown_group"
UNKNOWN_USER = "unknown_user"
UNKNOWN_CHAT = "group"
SUMMARY_CURSOR_PATH = "data/summary_cursor.json"

SUMMARY_AGENT_CONFIG = load_current_agent_config(__file__)


try:
    DEFAULT_MAX_LINE_CHARS = int(SUMMARY_AGENT_CONFIG.get("max_line_chars", 300))
except (TypeError, ValueError):
    DEFAULT_MAX_LINE_CHARS = 300

try:
    DEFAULT_MAX_LINES = int(SUMMARY_AGENT_CONFIG.get("max_lines", 500))
except (TypeError, ValueError):
    DEFAULT_MAX_LINES = 500

DEFAULT_LLM_MODEL = str(SUMMARY_AGENT_CONFIG.get("model") or "gpt-4o-mini")
try:
    DEFAULT_LLM_TEMPERATURE = float(SUMMARY_AGENT_CONFIG.get("temperature", 0.2))
except (TypeError, ValueError):
    DEFAULT_LLM_TEMPERATURE = 0.2

DEFAULT_SUMMARY_GLOBAL_OVERVIEW = bool(SUMMARY_AGENT_CONFIG.get("summary_global_overview", False))
DEFAULT_SUMMARY_SEND_MODE = str(SUMMARY_AGENT_CONFIG.get("summary_send_mode") or "single_message").strip().lower()


HEADER_RE = re.compile(
    r"^(?:\[chat:(?P<chat_type>[^\]]+)\])?\[group:(?P<group_id>[^\]]+)\]\[user:(?P<user_id>[^\]]+)\](?:\[name:(?P<user_name>[^\]]+)\])?$"
)
TIME_PREFIX_RE = re.compile(r"^\[(?P<hhmm>\d{2}:\d{2})\]\s*")


SYSTEM_SUMMARY_PROMPT = str(SUMMARY_AGENT_CONFIG.get("system_prompt") or """你是资深项目管理助理，负责将群聊消息总结成可执行日报。

# 任务目标
- 从输入消息中提炼：整体进展、关键要点、风险、待办。
- 输出必须可直接供管理者决策与跟进。

# 事实约束（必须遵守）
- 只能依据输入消息，不得补充外部事实，不得猜测。
- 信息不足时，允许留空列表或使用保守表述。
- 不复述闲聊和寒暄，优先保留可执行信息。

# 质量标准
- highlights: 3~6 条，聚焦里程碑、决策、结果变化。
- risks: 0~5 条，描述阻塞/不确定性及潜在影响。
- todos: 0~5 条，写成可执行动作（尽量含对象或时间线索）。
- evidence: 0~6 条，摘取支撑结论的短证据（简短原句或片段）。
- overview: 1 句中文，先给全局结论，再给态势判断。

# 输出要求
- 使用中文，简洁、客观、无修辞。
- 严格按结构化字段返回，不要输出多余说明。""")


USER_SUMMARY_PROMPT_TEMPLATE = str(SUMMARY_AGENT_CONFIG.get("user_prompt_template") or """请总结以下单个 chunk 的消息。
chunk_index: {chunk_index}
source_count: {source_count}
sources: {sources}
source_details:
{source_details}
unique_lines: {unique_lines}

输入消息（每行格式: group|user:[HH:MM] message）：
---BEGIN_MESSAGES---
{payload_text}
---END_MESSAGES---

请基于以上输入，按约定结构化字段输出总结。""")


GROUP_REDUCE_SYSTEM_PROMPT = str(SUMMARY_AGENT_CONFIG.get("group_reduce_system_prompt") or """你是项目总结整合助手。
你会拿到同一个群的多个分块摘要，请将它们整合为该群的单份最终摘要。

要求：
1) 只能基于输入，不编造事实。
2) 去重并合并同类项，保留最关键、可执行的信息。
3) highlights 控制在 3~6 条，risks/todos 各 0~5 条。
4) overview 用一句中文概括该群今天最重要进展。""")


GROUP_REDUCE_USER_PROMPT_TEMPLATE = str(SUMMARY_AGENT_CONFIG.get("group_reduce_user_prompt_template") or """请整合以下同一会话来源的分块摘要。
chat_type: {chat_type}
group_id: {group_id}
chunk_count: {chunk_count}

---BEGIN_CHUNK_SUMMARIES---
{chunk_summaries}
---END_CHUNK_SUMMARIES---

请按结构化字段返回该会话的最终摘要。""")


GLOBAL_OVERVIEW_SYSTEM_PROMPT = str(SUMMARY_AGENT_CONFIG.get("global_overview_system_prompt") or """你是日报总览助手。
请基于“各群摘要”生成一段全局总览，用 1~2 句话概括当天整体态势与重点。""")


GLOBAL_OVERVIEW_USER_PROMPT_TEMPLATE = str(SUMMARY_AGENT_CONFIG.get("global_overview_user_prompt_template") or """请阅读以下各群摘要并输出全局总览。

---BEGIN_GROUP_SUMMARIES---
{group_summaries}
---END_GROUP_SUMMARIES---""")


@dataclass
class SummaryBlock:
    """单个来源分组块。

    该结构直接承载来源上下文：
    - chat_type: 消息来源类型（group/private）
    - group_id: 群来源（私聊时为 private）
    - user_id: 发送者来源
    - user_name: 发送者昵称（群名片优先）
    - lines: 该来源块下的原始行
    """

    chat_type: str
    group_id: str
    user_id: str
    lines: list[str]
    user_name: str = UNKNOWN_USER


@dataclass
class SummaryChunk:
    """summary 预处理结果载体。

    角色：
    - preprocess_summary_chunk 的标准输出
    - 上接原始文本解析，下接 workflow payload 组装
    """

    raw_text: str
    blocks: list[SummaryBlock]
    texts: list[str]
    text: str
    total_lines: int
    non_empty_lines: int
    unique_lines: int
    block_count: int
    output_chars: int
    elapsed_ms: float


@dataclass
class SummaryWorkflowPayload:
    """给后续 summary agent 的标准输入载体。

    关系：
    - lines/text 来自 SummaryChunk.texts/text
    - blocks 来自 SummaryChunk.blocks
    - stats 由 SummaryChunk 派生
    """

    lines: list[str]
    text: str
    blocks: list[SummaryBlock]
    stats: dict[str, float | int]


@dataclass
class SummaryMapResult:
    """单个 chunk 的摘要结果（Map 阶段产物）。"""

    chunk_index: int
    blocks: list[SummaryBlock] = field(default_factory=list)
    overview: str = ""
    highlights: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class SummaryFinalResult:
    """全量汇总结果（最终发送前结构）。"""

    date: str = ""
    overview: str = ""
    highlights: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    chunk_count: int = 0
    message_count: int = 0
    sources: list[str] = field(default_factory=list)
    trace_lines: list[str] = field(default_factory=list)
    map_results: list[SummaryMapResult] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass
class GroupSummaryResult:
    """单个群（或私聊会话）的最终摘要结果。"""

    chat_type: str
    group_id: str
    summary: SummaryFinalResult


@dataclass
class GroupedSummaryResult:
    """多群聚合摘要结果。"""

    date: str = ""
    group_results: list[GroupSummaryResult] = field(default_factory=list)
    global_overview: str = ""
    chunk_count: int = 0
    message_count: int = 0
    elapsed_ms: float = 0.0


class ChunkSummarySchema(BaseModel):
    """LLM 结构化输出 Schema（Map 节点）。"""

    overview: str = Field(default="")
    highlights: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    todos: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class GlobalOverviewSchema(BaseModel):
    overview: str = Field(default="")


class SummaryGraphState(TypedDict):
    """LangGraph 状态机。"""

    payload: SummaryWorkflowPayload
    chunk_index: int
    llm_calls: int
    map_result: SummaryMapResult | None
    final_result: SummaryFinalResult | None


def build_summary_chunks_from_log_lines(
    lines: list[str],
    *,
    chunk_size: int,
    run_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """从日志原始行构建 summary chunks（解析+筛选+分块）。"""
    records: list[dict[str, str]] = []
    for line in lines:
        group_id, user_id, user_name, ts, message, chat_type = parse_summary_log_line(line)
        records.append(
            {
                "group_id": group_id,
                "user_id": user_id,
                "user_name": user_name,
                "ts": ts,
                "message": message,
                "chat_type": chat_type,
            }
        )

    filtered_records, meta = filter_records_for_summary(records, run_mode=run_mode)
    if not filtered_records:
        return [], meta

    grouped_messages: dict[tuple[str, str, str, str], list[tuple[str, str]]] = {}
    for record in filtered_records:
        chat_type = _normalize_chat_type(record.get("chat_type", "group"))
        group_id = str(record.get("group_id", UNKNOWN_GROUP))
        user_id = str(record.get("user_id", UNKNOWN_USER))
        user_name = str(record.get("user_name", UNKNOWN_USER))
        ts = str(record.get("ts", ""))
        message = str(record.get("message", "")).strip()
        if not message:
            continue
        key = (chat_type, group_id, user_id, user_name)
        if key not in grouped_messages:
            grouped_messages[key] = []
        grouped_messages[key].append((ts, message))

    grouped_chunks_by_group: dict[tuple[str, str], list[str]] = {}
    for (chat_type, group_id, user_id, user_name), messages in grouped_messages.items():
        source_chunks = _split_source_messages(
            chat_type=chat_type,
            group_id=group_id,
            user_id=user_id,
            user_name=user_name,
            messages=messages,
            chunk_size=chunk_size,
        )
        group_key = (chat_type, group_id)
        if group_key not in grouped_chunks_by_group:
            grouped_chunks_by_group[group_key] = []
        grouped_chunks_by_group[group_key].extend(source_chunks)

    group_jobs: list[dict[str, Any]] = []
    merged_chunk_total = 0
    for chat_type, group_id in sorted(grouped_chunks_by_group.keys()):
        merged_chunks = _merge_small_chunks(grouped_chunks_by_group[(chat_type, group_id)], chunk_size)
        if not merged_chunks:
            continue
        group_jobs.append(
            {
                "chat_type": chat_type,
                "group_id": group_id,
                "chunks": merged_chunks,
            }
        )
        merged_chunk_total += len(merged_chunks)

    if meta.get("run_mode") == "manual" and meta.get("cursor_key") and meta.get("cursor_after"):
        save_summary_cursor(str(meta["cursor_key"]), str(meta["cursor_after"]))

    meta["group_count"] = str(len(group_jobs))
    meta["group_chunks"] = str(sum(len(chunks) for chunks in grouped_chunks_by_group.values()))
    meta["final_chunks"] = str(merged_chunk_total)
    return group_jobs, meta


def filter_records_for_summary(
    records: list[dict[str, str]],
    *,
    run_mode: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """按 chat scope + cursor 筛选日志记录。"""
    scope = _normalize_scope(str(SUMMARY_AGENT_CONFIG.get("summary_chat_scope") or "group"))
    group_filter_mode = _normalize_group_filter_mode(
        str(SUMMARY_AGENT_CONFIG.get("summary_group_filter_mode") or "all")
    )
    configured_group_ids = SUMMARY_AGENT_CONFIG.get("summary_group_ids")
    group_id_set = {
        str(item).strip()
        for item in (configured_group_ids if isinstance(configured_group_ids, list) else [])
        if str(item).strip()
    }
    cursor_key = f"manual_{scope}"
    cursor_before = ""
    normalized_run_mode = str(run_mode or "manual").strip().lower()
    if normalized_run_mode not in {"manual", "auto"}:
        normalized_run_mode = "manual"

    use_cursor = normalized_run_mode == "manual"
    today_str = datetime.now().astimezone().strftime("%Y-%m-%d")

    if use_cursor:
        cursor_before = load_summary_cursor(cursor_key)

    cursor_dt = _parse_iso_dt(cursor_before) if cursor_before else None
    filtered_records: list[dict[str, str]] = []
    latest_dt = cursor_dt

    for record in records:
        chat_type = _normalize_chat_type(record.get("chat_type", "group"))
        if scope != "all" and chat_type != scope:
            continue

        group_id = str(record.get("group_id", UNKNOWN_GROUP)).strip() or UNKNOWN_GROUP
        if chat_type == "group" and group_id_set:
            if group_filter_mode == "include" and group_id not in group_id_set:
                continue
            if group_filter_mode == "exclude" and group_id in group_id_set:
                continue

        ts = str(record.get("ts", "")).strip()
        current_dt = _parse_iso_dt(ts)
        if normalized_run_mode == "auto":
            if current_dt is None or current_dt.astimezone().strftime("%Y-%m-%d") != today_str:
                continue

        if use_cursor and cursor_dt is not None:
            if current_dt is None or current_dt <= cursor_dt:
                continue

        if current_dt is not None and (latest_dt is None or current_dt > latest_dt):
            latest_dt = current_dt

        normalized = dict(record)
        normalized["chat_type"] = chat_type
        normalized["group_id"] = group_id
        normalized["message"] = str(record.get("message", "")).strip()
        filtered_records.append(normalized)

    cursor_after = ""
    if latest_dt is not None:
        cursor_after = latest_dt.isoformat(timespec="seconds")
    elif cursor_before:
        cursor_after = cursor_before

    return filtered_records, {
        "run_mode": normalized_run_mode,
        "scope": scope,
        "cursor_key": cursor_key,
        "cursor_before": cursor_before,
        "cursor_after": cursor_after,
    }


def parse_summary_log_line(line: str) -> tuple[str, str, str, str, str, str]:
    """解析 summary 日志行，兼容 JSONL 与旧文本格式。"""
    try:
        record = json.loads(line)
        if isinstance(record, dict):
            return (
                str(record.get("group_id", UNKNOWN_GROUP)),
                str(record.get("user_id", UNKNOWN_USER)),
                str(record.get("user_name", UNKNOWN_USER)),
                str(record.get("ts", "")),
                str(record.get("cleaned_message", record.get("raw_message", ""))).strip(),
                _normalize_chat_type(record.get("chat_type", "group")),
            )
    except json.JSONDecodeError:
        pass

    if ":" not in line:
        return UNKNOWN_GROUP, UNKNOWN_USER, UNKNOWN_USER, "", line.strip(), "group"

    prefix, message = line.split(":", 1)
    if "|" in prefix:
        group_id, user_id = prefix.split("|", 1)
        normalized_user_id = user_id.strip() or UNKNOWN_USER
        return group_id.strip() or UNKNOWN_GROUP, normalized_user_id, normalized_user_id, "", message.strip(), "group"

    return prefix.strip() or UNKNOWN_GROUP, UNKNOWN_USER, UNKNOWN_USER, "", message.strip(), "group"


def load_summary_cursor(cursor_key: str) -> str:
    if not os.path.exists(SUMMARY_CURSOR_PATH):
        return ""
    try:
        with open(SUMMARY_CURSOR_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get(cursor_key, "")).strip()


def save_summary_cursor(cursor_key: str, ts: str) -> None:
    os.makedirs(os.path.dirname(SUMMARY_CURSOR_PATH), exist_ok=True)
    payload: dict[str, Any] = {}
    if os.path.exists(SUMMARY_CURSOR_PATH):
        try:
            with open(SUMMARY_CURSOR_PATH, "r", encoding="utf-8") as file:
                existing = json.load(file)
            if isinstance(existing, dict):
                payload.update(existing)
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload[cursor_key] = ts
    with open(SUMMARY_CURSOR_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def preprocess_summary_chunk(
    raw_message: str,
    *,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> SummaryChunk:
    """预处理 SUMMARY 队列消息，兼容 scheduler 的分组头部格式。"""
    started_at = perf_counter()
    raw_text = raw_message if isinstance(raw_message, str) else str(raw_message or "")
    raw_lines = raw_text.splitlines()
    blocks = _parse_blocks(raw_lines)

    normalized_lines, non_empty_lines = _collect_normalized_lines(
        blocks=blocks,
        max_line_chars=max_line_chars,
        max_lines=max_lines,
    )

    output_text = "\n".join(normalized_lines)
    elapsed_ms = (perf_counter() - started_at) * 1000
    return SummaryChunk(
        raw_text=raw_text,
        blocks=blocks,
        texts=normalized_lines,
        text=output_text,
        total_lines=len(raw_lines),
        non_empty_lines=non_empty_lines,
        unique_lines=len(normalized_lines),
        block_count=len(blocks),
        output_chars=len(output_text),
        elapsed_ms=elapsed_ms,
    )


def prepare_summary_payload(
    raw_message: str,
    *,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> SummaryWorkflowPayload:
    """summary 工作流入口（agent 前置）：返回统一 payload。"""
    chunk = preprocess_summary_chunk(
        raw_message,
        max_line_chars=max_line_chars,
        max_lines=max_lines,
    )
    return SummaryWorkflowPayload(
        lines=chunk.texts,
        text=chunk.text,
        blocks=chunk.blocks,
        stats={
            "total_lines": chunk.total_lines,
            "non_empty_lines": chunk.non_empty_lines,
            "unique_lines": chunk.unique_lines,
            "block_count": chunk.block_count,
            "output_chars": chunk.output_chars,
            "elapsed_ms": chunk.elapsed_ms,
        },
    )


def run_summary_graph(
    raw_message: str,
    *,
    chunk_index: int = 1,
    model_name: str | None = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
) -> SummaryFinalResult:
    """运行 summary 的 LangGraph 核心流程（当前为最少节点两步）。

    节点设计：
    - map_node: 一次 LLM 调用，输出 SummaryMapResult
    - finalize_node: 无 LLM 调用，组装 SummaryFinalResult
    """
    workflow_started_at = perf_counter()
    payload = prepare_summary_payload(raw_message)

    graph = _build_summary_graph(model_name=model_name, temperature=temperature)
    result = graph.invoke(
        {
            "payload": payload,
            "chunk_index": chunk_index,
            "llm_calls": 0,
            "map_result": None,
            "final_result": None,
        }
    )

    final_result = result.get("final_result")
    if not isinstance(final_result, SummaryFinalResult):
        raise RuntimeError("Summary graph 执行失败：未生成 final_result")

    final_result.elapsed_ms = (perf_counter() - workflow_started_at) * 1000
    return final_result


def run_grouped_summary_graph(
    group_jobs: list[dict[str, Any]],
    *,
    model_name: str | None = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
) -> GroupedSummaryResult:
    """按群执行 summary（每群可含多个 chunk），并聚合为单次输出结果。"""
    started_at = perf_counter()
    today_text = datetime.now().strftime("%Y-%m-%d")
    group_results: list[GroupSummaryResult] = []
    total_chunks = 0
    total_messages = 0
    llm = _build_llm(model_name=model_name, temperature=temperature)
    structured_chunk_reducer = llm.with_structured_output(ChunkSummarySchema)

    for job in group_jobs:
        if not isinstance(job, dict):
            continue

        chat_type = _normalize_chat_type(job.get("chat_type", "group"))
        group_id = str(job.get("group_id", UNKNOWN_GROUP)).strip() or UNKNOWN_GROUP
        raw_chunks = job.get("chunks")
        chunk_texts = [str(item) for item in raw_chunks] if isinstance(raw_chunks, list) else []
        chunk_texts = [item for item in chunk_texts if item.strip()]
        if not chunk_texts:
            continue

        chunk_results: list[SummaryFinalResult] = []
        for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
            chunk_results.append(
                run_summary_graph(
                    chunk_text,
                    chunk_index=chunk_index,
                    model_name=model_name,
                    temperature=temperature,
                )
            )

        merged_sources: list[str] = []
        merged_trace_lines: list[str] = []
        merged_map_results: list[SummaryMapResult] = []
        chunk_summary_lines: list[str] = []
        for idx, chunk_result in enumerate(chunk_results, start=1):
            merged_sources.extend(chunk_result.sources)
            merged_trace_lines.extend(chunk_result.trace_lines)
            merged_map_results.extend(chunk_result.map_results)
            chunk_summary_lines.append(
                "\n".join(
                    [
                        f"chunk#{idx}",
                        f"overview: {chunk_result.overview or '（无）'}",
                        "highlights:",
                        *[f"- {item}" for item in _safe_list(chunk_result.highlights, max_items=10)],
                        "risks:",
                        *[f"- {item}" for item in _safe_list(chunk_result.risks, max_items=10)],
                        "todos:",
                        *[f"- {item}" for item in _safe_list(chunk_result.todos, max_items=10)],
                    ]
                )
            )

        if len(chunk_results) == 1:
            reduced_overview = chunk_results[0].overview
            reduced_highlights = chunk_results[0].highlights
            reduced_risks = chunk_results[0].risks
            reduced_todos = chunk_results[0].todos
        else:
            reduce_messages = [
                SystemMessage(content=GROUP_REDUCE_SYSTEM_PROMPT),
                HumanMessage(
                    content=GROUP_REDUCE_USER_PROMPT_TEMPLATE.format(
                        chat_type=chat_type,
                        group_id=group_id,
                        chunk_count=len(chunk_results),
                        chunk_summaries="\n\n".join(chunk_summary_lines),
                    )
                ),
            ]
            reduce_result = structured_chunk_reducer.invoke(reduce_messages)
            reduced_overview = (reduce_result.overview or "").strip() or "今日暂无可总结内容。"
            reduced_highlights = _safe_list(reduce_result.highlights, max_items=6)
            reduced_risks = _safe_list(reduce_result.risks, max_items=5)
            reduced_todos = _safe_list(reduce_result.todos, max_items=5)

        group_summary = SummaryFinalResult(
            date=today_text,
            overview=reduced_overview,
            highlights=_safe_list(reduced_highlights, max_items=6),
            risks=_safe_list(reduced_risks, max_items=5),
            todos=_safe_list(reduced_todos, max_items=5),
            chunk_count=len(chunk_results),
            message_count=sum(item.message_count for item in chunk_results),
            sources=_safe_list(merged_sources, max_items=200),
            trace_lines=_safe_list(merged_trace_lines, max_items=20),
            map_results=merged_map_results,
        )
        group_results.append(
            GroupSummaryResult(
                chat_type=chat_type,
                group_id=group_id,
                summary=group_summary,
            )
        )
        total_chunks += group_summary.chunk_count
        total_messages += group_summary.message_count

    global_overview = ""
    if DEFAULT_SUMMARY_GLOBAL_OVERVIEW and group_results:
        try:
            structured_overview_llm = llm.with_structured_output(GlobalOverviewSchema)
            group_summary_text = "\n\n".join(
                [
                    "\n".join(
                        [
                            f"group={item.group_id}, chat_type={item.chat_type}",
                            f"overview: {item.summary.overview or '（无）'}",
                            "highlights:",
                            *[f"- {h}" for h in _safe_list(item.summary.highlights, max_items=8)],
                            "risks:",
                            *[f"- {r}" for r in _safe_list(item.summary.risks, max_items=8)],
                            "todos:",
                            *[f"- {t}" for t in _safe_list(item.summary.todos, max_items=8)],
                        ]
                    )
                    for item in group_results
                ]
            )
            global_messages = [
                SystemMessage(content=GLOBAL_OVERVIEW_SYSTEM_PROMPT),
                HumanMessage(content=GLOBAL_OVERVIEW_USER_PROMPT_TEMPLATE.format(group_summaries=group_summary_text)),
            ]
            overview_result = structured_overview_llm.invoke(global_messages)
            global_overview = (overview_result.overview or "").strip()
        except Exception:
            global_overview = ""

    return GroupedSummaryResult(
        date=today_text,
        group_results=group_results,
        global_overview=global_overview,
        chunk_count=total_chunks,
        message_count=total_messages,
        elapsed_ms=(perf_counter() - started_at) * 1000,
    )


def format_summary_message(result: SummaryFinalResult) -> str:
    """将最终结果格式化为可直接发送给 QQ 的文本。"""
    date_text = result.date or datetime.now().strftime("%Y-%m-%d")
    highlights = _safe_list(result.highlights, max_items=6)
    risks = _safe_list(result.risks, max_items=5)
    todos = _safe_list(result.todos, max_items=5)
    trace_lines = _safe_list(result.trace_lines, max_items=20)
    if not trace_lines and result.sources:
        trace_lines = [f"来源标识：{', '.join(result.sources)}"]

    highlight_text = "\n".join(f"- {item}" for item in highlights) or "- （暂无）"
    risk_text = "\n".join(f"- {item}" for item in risks) or "- （暂无）"
    todo_text = "\n".join(f"- {item}" for item in todos) or "- （暂无）"
    trace_text = "\n".join(f"- {item}" for item in trace_lines) or "- （暂无）"

    return (
        f"【每日总结 {date_text}】\n"
        f"概览：{result.overview or '（暂无概览）'}\n\n"
        f"【溯源】\n{trace_text}\n\n"
        f"【关键要点】\n{highlight_text}\n\n"
        f"【风险】\n{risk_text}\n\n"
        f"【待办】\n{todo_text}\n\n"
        f"（chunks={result.chunk_count}, messages={result.message_count}, sources={len(result.sources)}, elapsed={result.elapsed_ms:.0f}ms）"
    )


def format_grouped_summary_message(result: GroupedSummaryResult) -> str:
    """将按群聚合后的 summary 结果格式化为私聊可读文本。"""
    date_text = result.date or datetime.now().strftime("%Y-%m-%d")
    if not result.group_results:
        return f"【每日总结 {date_text}】\n今日暂无可总结内容。"

    lines: list[str] = [f"【每日总结 {date_text}】"]
    if result.global_overview.strip():
        lines.append(f"全局总览：{result.global_overview.strip()}")
    for group_result in result.group_results:
        summary = group_result.summary
        source_label = (
            f"私聊({group_result.group_id})"
            if group_result.chat_type == "private"
            else f"{group_result.group_id}群"
        )

        highlights = _safe_list(summary.highlights, max_items=6)
        risks = _safe_list(summary.risks, max_items=5)
        todos = _safe_list(summary.todos, max_items=5)
        trace_lines = _safe_list(summary.trace_lines, max_items=5)

        highlight_text = "\n".join(f"- {item}" for item in highlights) or "- （暂无）"
        risk_text = "\n".join(f"- {item}" for item in risks) or "- （暂无）"
        todo_text = "\n".join(f"- {item}" for item in todos) or "- （暂无）"
        trace_text = "\n".join(f"- {item}" for item in trace_lines) or "- （暂无）"

        lines.append(
            f"\n【{source_label}】\n"
            f"概览：{summary.overview or '（暂无概览）'}\n"
            f"溯源：\n{trace_text}\n"
            f"关键要点：\n{highlight_text}\n"
            f"风险：\n{risk_text}\n"
            f"待办：\n{todo_text}\n"
            f"（chunks={summary.chunk_count}, messages={summary.message_count}, sources={len(summary.sources)}）"
        )

    lines.append(
        f"\n（groups={len(result.group_results)}, total_chunks={result.chunk_count}, total_messages={result.message_count}, elapsed={result.elapsed_ms:.0f}ms）"
    )
    return "\n".join(lines)


def format_grouped_summary_messages(result: GroupedSummaryResult) -> list[str]:
    """多消息发送模式：返回消息列表（先全局，再逐群）。"""
    date_text = result.date or datetime.now().strftime("%Y-%m-%d")
    if not result.group_results:
        return [f"【每日总结 {date_text}】\n今日暂无可总结内容。"]

    messages: list[str] = [
        f"【每日总结 {date_text}】\n"
        f"{('全局总览：' + result.global_overview) if result.global_overview.strip() else '（未启用全局总览）'}\n"
        f"（groups={len(result.group_results)}, total_chunks={result.chunk_count}, total_messages={result.message_count}）"
    ]
    for group_result in result.group_results:
        summary = group_result.summary
        source_label = (
            f"私聊({group_result.group_id})"
            if group_result.chat_type == "private"
            else f"{group_result.group_id}群"
        )
        highlights = _safe_list(summary.highlights, max_items=6)
        risks = _safe_list(summary.risks, max_items=5)
        todos = _safe_list(summary.todos, max_items=5)
        trace_lines = _safe_list(summary.trace_lines, max_items=5)
        trace_text = "\n".join(f"- {item}" for item in trace_lines) if trace_lines else "- （暂无）"
        highlight_text = "\n".join(f"- {item}" for item in highlights) if highlights else "- （暂无）"
        risk_text = "\n".join(f"- {item}" for item in risks) if risks else "- （暂无）"
        todo_text = "\n".join(f"- {item}" for item in todos) if todos else "- （暂无）"
        messages.append(
            f"【{source_label}】\n"
            f"概览：{summary.overview or '（暂无概览）'}\n"
            f"溯源：\n{trace_text}\n"
            f"关键要点：\n{highlight_text}\n"
            f"风险：\n{risk_text}\n"
            f"待办：\n{todo_text}\n"
            f"（chunks={summary.chunk_count}, messages={summary.message_count}, sources={len(summary.sources)}）"
        )
    return messages


def get_summary_send_mode() -> str:
    mode = (DEFAULT_SUMMARY_SEND_MODE or "single_message").strip().lower()
    if mode in {"single_message", "multi_message"}:
        return mode
    return "single_message"


def _build_summary_graph(*, model_name: str | None, temperature: float):
    """
    Agent核心流程Graph构建
    """

    llm = _build_llm(model_name=model_name, temperature=temperature)

    def map_node(state: SummaryGraphState) -> dict[str, Any]:
        payload = state["payload"]
        chunk_index = state["chunk_index"]

        if not payload.lines:
            empty_result = SummaryMapResult(
                chunk_index=chunk_index,
                blocks=payload.blocks,
                overview="今日暂无可总结内容。",
                highlights=[],
                risks=[],
                todos=[],
                evidence=[],
            )
            return {"map_result": empty_result, "llm_calls": state["llm_calls"]}

        structured_llm = llm.with_structured_output(ChunkSummarySchema)
        source_refs, source_details, _trace_lines = _analyze_blocks(payload.blocks)
        messages = [
            SystemMessage(
                content=SYSTEM_SUMMARY_PROMPT
            ),
            HumanMessage(
                content=USER_SUMMARY_PROMPT_TEMPLATE.format(
                    chunk_index=chunk_index,
                    source_count=len(source_refs),
                    sources=", ".join(source_refs) if source_refs else "(none)",
                    source_details="\n".join(source_details) if source_details else "(none)",
                    unique_lines=payload.stats.get("unique_lines", 0),
                    payload_text=payload.text,
                )
            ),
        ]

        llm_result = structured_llm.invoke(messages)
        map_result = SummaryMapResult(
            chunk_index=chunk_index,
            blocks=payload.blocks,
            overview=(llm_result.overview or "").strip(),
            highlights=_safe_list(llm_result.highlights, min_items=3, max_items=6),
            risks=_safe_list(llm_result.risks, max_items=5),
            todos=_safe_list(llm_result.todos, max_items=5),
            evidence=_safe_list(llm_result.evidence, max_items=6),
        )
        return {"map_result": map_result, "llm_calls": state["llm_calls"] + 1}

    def finalize_node(state: SummaryGraphState) -> dict[str, Any]:
        payload = state["payload"]
        map_result = state.get("map_result")
        if map_result is None:
            map_result = SummaryMapResult(
                chunk_index=state["chunk_index"],
                blocks=payload.blocks,
                overview="今日暂无可总结内容。",
            )

        source_refs, _source_details, trace_lines = _analyze_blocks(payload.blocks)
        final_result = SummaryFinalResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            overview=map_result.overview,
            highlights=map_result.highlights,
            risks=map_result.risks,
            todos=map_result.todos,
            chunk_count=1,
            message_count=int(payload.stats.get("unique_lines", 0)),
            sources=source_refs,
            trace_lines=trace_lines,
            map_results=[map_result],
        )
        return {"final_result": final_result}

    graph = StateGraph(SummaryGraphState)
    graph.add_node("map_node", map_node)
    graph.add_node("finalize_node", finalize_node)
    graph.add_edge(START, "map_node")
    graph.add_edge("map_node", "finalize_node")
    graph.add_edge("finalize_node", END)
    return graph.compile()


def _build_llm(*, model_name: str | None, temperature: float) -> ChatOpenAI:
    """
    给Agent接入LLM
    """
    if load_dotenv is not None:
        load_dotenv(override=False)

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("缺少 LLM_API_KEY，请在 .env 中配置")

    base_url = os.getenv("LLM_API_BASE_URL")
    resolved_model = model_name or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL

    llm_kwargs: dict[str, Any] = {
        "model_name": resolved_model,
        "temperature": temperature,
        "openai_api_key": api_key,
    }
    if base_url:
        llm_kwargs["openai_api_base"] = base_url

    return ChatOpenAI(**llm_kwargs)


def _collect_normalized_lines(
    *,
    blocks: list[SummaryBlock],
    max_line_chars: int,
    max_lines: int,
) -> tuple[list[str], int]:
    deduped_lines: list[str] = []
    seen: set[str] = set()
    non_empty_lines = 0

    for block in blocks:
        for line in block.lines:
            cleaned = line.strip()
            if not cleaned:
                continue

            non_empty_lines += 1
            if len(cleaned) > max_line_chars:
                cleaned = cleaned[: max_line_chars - 1] + "…"
            normalized = f"{block.group_id}|{block.user_id}:{cleaned}"

            if normalized in seen:
                continue

            seen.add(normalized)
            deduped_lines.append(normalized)

            if len(deduped_lines) >= max_lines:
                return deduped_lines, non_empty_lines

    return deduped_lines, non_empty_lines


def _parse_blocks(lines: list[str]) -> list[SummaryBlock]:
    blocks: list[SummaryBlock] = []
    current_chat = UNKNOWN_CHAT
    current_group = UNKNOWN_GROUP
    current_user = UNKNOWN_USER
    current_user_name = UNKNOWN_USER
    current_lines: list[str] = []
    saw_content = False

    def flush() -> None:
        nonlocal current_lines
        blocks.append(
            SummaryBlock(
                chat_type=current_chat,
                group_id=current_group,
                user_id=current_user,
                user_name=current_user_name,
                lines=current_lines,
            )
        )
        current_lines = []

    for line in lines:
        header = HEADER_RE.match(line.strip())
        if header:
            if saw_content:
                flush()
            parsed_chat = (header.group("chat_type") or "").strip().lower()
            current_chat = _normalize_chat_type(parsed_chat)
            current_group = header.group("group_id").strip() or UNKNOWN_GROUP
            current_user = header.group("user_id").strip() or UNKNOWN_USER
            parsed_user_name = (header.group("user_name") or "").strip()
            current_user_name = parsed_user_name or current_user
            current_lines = []
            saw_content = True
            continue

        if not saw_content:
            current_chat = UNKNOWN_CHAT
            current_group = UNKNOWN_GROUP
            current_user = UNKNOWN_USER
            current_user_name = UNKNOWN_USER
            current_lines = []
            saw_content = True

        current_lines.append(line)

    if saw_content:
        flush()

    return blocks


def _analyze_blocks(blocks: list[SummaryBlock]) -> tuple[list[str], list[str], list[str]]:
    """单次遍历 blocks，统一生成来源引用/细节/溯源文案。"""
    source_ref_set: set[str] = set()
    source_details: list[str] = []
    group_summary: dict[str, dict[str, Any]] = {}

    for block in blocks:
        chat_type = _normalize_chat_type(block.chat_type)
        group_id = block.group_id or UNKNOWN_GROUP
        user_id = block.user_id or UNKNOWN_USER
        user_name = (block.user_name or "").strip() or user_id

        source_ref_set.add(f"{chat_type}|{group_id}|{user_id}")
        source_details.append(
            f"- chat={chat_type}, group={group_id}, user={user_id}, name={user_name}, lines={len(block.lines)}"
        )

        group_key = f"{chat_type}:{group_id}"
        group_entry = group_summary.setdefault(
            group_key,
            {
                "chat_type": chat_type,
                "group_id": group_id,
                "senders": set(),
                "times": [],
                "message_count": 0,
            },
        )
        sender_ref = f"{user_name}({user_id})" if user_id else user_name
        group_entry["senders"].add(sender_ref)

        for line in block.lines:
            cleaned = line.strip()
            if not cleaned:
                continue
            group_entry["message_count"] += 1

            match = TIME_PREFIX_RE.match(cleaned)
            hhmm = match.group("hhmm") if match else ""
            if hhmm:
                group_entry["times"].append(hhmm)

    trace_lines: list[str] = []
    for group_key in sorted(group_summary.keys()):
        entry = group_summary[group_key]
        senders = sorted(entry["senders"])
        if not senders:
            sender_text = "未知发送者"
        elif len(senders) <= 3:
            sender_text = "、".join(senders)
        else:
            sender_text = f"{'、'.join(senders[:3])} 等{len(senders)}人"

        normalized_times = sorted({time_text for time_text in entry["times"] if time_text})
        if not normalized_times:
            time_text = "时间未知"
        elif len(normalized_times) == 1:
            time_text = f"今天{normalized_times[0]}"
        else:
            time_text = f"今天{normalized_times[0]}~{normalized_times[-1]}"

        source_label = (
            f"私聊({entry['group_id']})"
            if entry.get("chat_type") == "private"
            else f"{entry['group_id']}群"
        )
        trace_lines.append(
            f"{source_label}主要说了 {entry['message_count']} 条消息，来自 {sender_text}，发送时间为{time_text}"
        )

    source_refs = sorted(source_ref_set)
    if not trace_lines and source_refs:
        trace_lines = [f"来源标识：{', '.join(source_refs)}"]

    return source_refs, source_details, trace_lines


def _safe_list(items: list[str], *, min_items: int = 0, max_items: int = 999) -> list[str]:
    cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    deduped = list(dict.fromkeys(cleaned))
    if len(deduped) < min_items:
        return deduped
    return deduped[:max_items]


def _split_source_messages(
    *,
    chat_type: str,
    group_id: str,
    user_id: str,
    user_name: str,
    messages: list[tuple[str, str]],
    chunk_size: int,
) -> list[str]:
    safe_name = (user_name or UNKNOWN_USER).replace("]", "）")
    header = (
        f"[chat:{_normalize_chat_type(chat_type)}]"
        f"[group:{group_id}][user:{user_id}][name:{safe_name}]"
    )
    max_body_chars = max(1, chunk_size - len(header) - 1)
    chunks: list[str] = []
    current_body = ""

    for ts, message in messages:
        dt = _parse_iso_dt(ts)
        time_prefix = dt.strftime("%H:%M") if dt is not None else ""
        normalized_message = f"[{time_prefix}] {message}" if time_prefix else message
        remaining = normalized_message
        while remaining:
            room = max_body_chars if not current_body else max_body_chars - len(current_body) - 1
            if room <= 0:
                chunks.append(f"{header}\n{current_body}")
                current_body = ""
                continue

            part = remaining[:room]
            remaining = remaining[room:]
            current_body = part if not current_body else f"{current_body}\n{part}"

            if remaining:
                chunks.append(f"{header}\n{current_body}")
                current_body = ""

    if current_body:
        chunks.append(f"{header}\n{current_body}")
    if not chunks:
        chunks.append(header)
    return chunks


def _merge_small_chunks(grouped_chunks: list[str], chunk_size: int) -> list[str]:
    merged: list[str] = []
    current = ""
    for block in grouped_chunks:
        if not current:
            current = block
            continue
        candidate = f"{current}\n\n{block}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            merged.append(current)
            current = block
    if current:
        merged.append(current)
    return merged


def _normalize_chat_type(chat_type: Any) -> str:
    text = str(chat_type or "").strip().lower()
    if text in {"group", "private"}:
        return text
    return "group"


def _normalize_scope(scope: str) -> str:
    value = str(scope or "").strip().lower()
    if value in {"group", "private", "all"}:
        return value
    return "group"


def _normalize_group_filter_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    if value in {"all", "include", "exclude"}:
        return value
    return "all"


def _parse_iso_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        normalized = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return dt
    except ValueError:
        return None
