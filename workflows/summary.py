"""
Summary 工作流中的预处理与 LangGraph 汇总模块。

职责：
1) 解析 scheduler 生成的 `[group:...][user:...]` 来源块。
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
    -> _normalize_raw_message
    -> _parse_blocks
    -> _collect_normalized_lines
    -> _build_chunk
  -> prepare_summary_payload
    -> _build_stats
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
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


UNKNOWN_GROUP = "unknown_group"
UNKNOWN_USER = "unknown_user"
DEFAULT_MAX_LINE_CHARS = 300
DEFAULT_MAX_LINES = 500
DEFAULT_LLM_MODEL = "gpt-4o-mini"
HEADER_RE = re.compile(r"^\[group:(?P<group_id>[^\]]+)\]\[user:(?P<user_id>[^\]]+)\]$")


SYSTEM_SUMMARY_PROMPT = """你是资深项目管理助理，负责将群聊消息总结成可执行日报。

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
- 严格按结构化字段返回，不要输出多余说明。"""


USER_SUMMARY_PROMPT_TEMPLATE = """请总结以下单个 chunk 的消息。
chunk_index: {chunk_index}
source_count: {source_count}
sources: {sources}
unique_lines: {unique_lines}

输入消息（每行格式: group|user:message）：
---BEGIN_MESSAGES---
{payload_text}
---END_MESSAGES---

请基于以上输入，按约定结构化字段输出总结。"""


@dataclass
class SummaryBlock:
    """单个来源分组块。

    该结构直接承载来源上下文：
    - group_id: 群来源
    - user_id: 发送者来源
    - lines: 该来源块下的原始行
    """

    group_id: str
    user_id: str
    lines: list[str]


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
    map_results: list[SummaryMapResult] = field(default_factory=list)
    elapsed_ms: float = 0.0


class ChunkSummarySchema(BaseModel):
    """LLM 结构化输出 Schema（Map 节点）。"""

    overview: str = Field(default="")
    highlights: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    todos: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class SummaryGraphState(TypedDict):
    """LangGraph 状态机。"""

    payload: SummaryWorkflowPayload
    chunk_index: int
    llm_calls: int
    map_result: SummaryMapResult | None
    final_result: SummaryFinalResult | None


def preprocess_summary_chunk(
    raw_message: str,
    *,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> SummaryChunk:
    """预处理 SUMMARY 队列消息，兼容 scheduler 的分组头部格式。"""
    started_at = perf_counter()
    raw_text = _normalize_raw_message(raw_message)
    raw_lines = raw_text.splitlines()
    blocks = _parse_blocks(raw_lines)

    normalized_lines, non_empty_lines = _collect_normalized_lines(
        blocks=blocks,
        max_line_chars=max_line_chars,
        max_lines=max_lines,
    )

    return _build_chunk(
        raw_text=raw_text,
        blocks=blocks,
        deduped_lines=normalized_lines,
        total_lines=len(raw_lines),
        non_empty_lines=non_empty_lines,
        started_at=started_at,
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
        stats=_build_stats(chunk),
    )


def run_summary_graph(
    raw_message: str,
    *,
    chunk_index: int = 1,
    model_name: str | None = None,
    temperature: float = 0.2,
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


def format_summary_message(result: SummaryFinalResult) -> str:
    """将最终结果格式化为可直接发送给 QQ 的文本。"""
    date_text = result.date or datetime.now().strftime("%Y-%m-%d")
    highlights = _safe_list(result.highlights, max_items=6)
    risks = _safe_list(result.risks, max_items=5)
    todos = _safe_list(result.todos, max_items=5)

    highlight_text = "\n".join(f"- {item}" for item in highlights) or "- （暂无）"
    risk_text = "\n".join(f"- {item}" for item in risks) or "- （暂无）"
    todo_text = "\n".join(f"- {item}" for item in todos) or "- （暂无）"

    return (
        f"【每日总结 {date_text}】\n"
        f"概览：{result.overview or '（暂无概览）'}\n\n"
        f"【关键要点】\n{highlight_text}\n\n"
        f"【风险】\n{risk_text}\n\n"
        f"【待办】\n{todo_text}\n\n"
        f"（chunks={result.chunk_count}, messages={result.message_count}, sources={len(result.sources)}, elapsed={result.elapsed_ms:.0f}ms）"
    )


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
        source_refs = sorted(
            {
                f"{block.group_id}|{block.user_id}"
                for block in payload.blocks
                if block.group_id or block.user_id
            }
        )
        messages = [
            SystemMessage(
                content=SYSTEM_SUMMARY_PROMPT
            ),
            HumanMessage(
                content=USER_SUMMARY_PROMPT_TEMPLATE.format(
                    chunk_index=chunk_index,
                    source_count=len(source_refs),
                    sources=", ".join(source_refs) if source_refs else "(none)",
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

        source_set = {
            f"{block.group_id}|{block.user_id}"
            for block in payload.blocks
            if block.group_id or block.user_id
        }
        final_result = SummaryFinalResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            overview=map_result.overview,
            highlights=map_result.highlights,
            risks=map_result.risks,
            todos=map_result.todos,
            chunk_count=1,
            message_count=int(payload.stats.get("unique_lines", 0)),
            sources=sorted(source_set),
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
    _load_env_if_possible()

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


def _load_env_if_possible() -> None:
    """
    读取.env文件
    （其实这个函数根本没必要。AI写的）
    """
    if load_dotenv is not None:
        load_dotenv(override=False)


def _normalize_raw_message(raw_message: str) -> str:
    if isinstance(raw_message, str):
        return raw_message
    return str(raw_message or "")


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
            cleaned = _truncate_text(cleaned, max_line_chars)
            normalized = f"{block.group_id}|{block.user_id}:{cleaned}"

            if normalized in seen:
                continue

            seen.add(normalized)
            deduped_lines.append(normalized)

            if len(deduped_lines) >= max_lines:
                return deduped_lines, non_empty_lines

    return deduped_lines, non_empty_lines


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _parse_blocks(lines: list[str]) -> list[SummaryBlock]:
    blocks: list[SummaryBlock] = []
    current_group = UNKNOWN_GROUP
    current_user = UNKNOWN_USER
    current_lines: list[str] = []
    saw_content = False

    def flush() -> None:
        nonlocal current_lines
        blocks.append(
            SummaryBlock(
                group_id=current_group,
                user_id=current_user,
                lines=current_lines,
            )
        )
        current_lines = []

    for line in lines:
        header = HEADER_RE.match(line.strip())
        if header:
            if saw_content:
                flush()
            current_group = header.group("group_id").strip() or UNKNOWN_GROUP
            current_user = header.group("user_id").strip() or UNKNOWN_USER
            current_lines = []
            saw_content = True
            continue

        if not saw_content:
            current_group = UNKNOWN_GROUP
            current_user = UNKNOWN_USER
            current_lines = []
            saw_content = True

        current_lines.append(line)

    if saw_content:
        flush()

    return blocks


def _build_chunk(
    *,
    raw_text: str,
    blocks: list[SummaryBlock],
    deduped_lines: list[str],
    total_lines: int,
    non_empty_lines: int,
    started_at: float,
) -> SummaryChunk:
    output_text = "\n".join(deduped_lines)
    elapsed_ms = (perf_counter() - started_at) * 1000
    return SummaryChunk(
        raw_text=raw_text,
        blocks=blocks,
        texts=deduped_lines,
        text=output_text,
        total_lines=total_lines,
        non_empty_lines=non_empty_lines,
        unique_lines=len(deduped_lines),
        block_count=len(blocks),
        output_chars=len(output_text),
        elapsed_ms=elapsed_ms,
    )


def _build_stats(chunk: SummaryChunk) -> dict[str, float | int]:
    return {
        "total_lines": chunk.total_lines,
        "non_empty_lines": chunk.non_empty_lines,
        "unique_lines": chunk.unique_lines,
        "block_count": chunk.block_count,
        "output_chars": chunk.output_chars,
        "elapsed_ms": chunk.elapsed_ms,
    }


def _safe_list(items: list[str], *, min_items: int = 0, max_items: int = 999) -> list[str]:
    cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    deduped = list(dict.fromkeys(cleaned))
    if len(deduped) < min_items:
        return deduped
    return deduped[:max_items]
