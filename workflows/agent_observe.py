"""统一 Agent 观测日志。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
import json
import uuid

try:
    import fcntl  # Unix 文件锁，防止多进程并发写入时产生坏行
except Exception:  # pragma: no cover
    fcntl = None


AGENT_LOG_PATH = Path("logs/agent_events.jsonl")
_AGENT_LOG_LOCK = Lock()
_REQUIRED_EVENT_KEYS = {"event_ts", "run_id", "agent_name", "task_type", "stage"}


def generate_run_id() -> str:
    """生成一次 agent 执行链路的 run_id。"""
    return uuid.uuid4().hex[:16]


def observe_agent_event(
    *,
    agent_name: str,
    stage: str,
    run_id: str = "",
    task_type: str = "",
    chat_type: str = "",
    group_id: str = "",
    user_id: str = "",
    user_name: str = "",
    ts: str = "",
    latency_ms: float | None = None,
    decision: dict[str, Any] | None = None,
    error: str = "",
    extra: dict[str, Any] | None = None,
) -> str:
    """写入一条结构化 Agent 观测日志(JSONL)。"""
    final_run_id = str(run_id).strip() or generate_run_id()
    payload: dict[str, Any] = {
        "event_ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "run_id": final_run_id,
        "agent_name": str(agent_name),
        "task_type": str(task_type),
        "stage": str(stage),
        "chat_type": str(chat_type),
        "group_id": str(group_id),
        "user_id": str(user_id),
        "user_name": str(user_name),
        "message_ts": str(ts),
    }
    if latency_ms is not None:
        payload["latency_ms"] = round(float(latency_ms), 2)
    if decision is not None:
        payload["decision"] = decision
    if error:
        payload["error"] = str(error)
    if extra is not None:
        payload["extra"] = extra

    with _AGENT_LOG_LOCK:
        AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AGENT_LOG_PATH.open("a", encoding="utf-8") as file:
            if fcntl is not None:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            try:
                file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                file.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    return final_run_id


def bind_agent_event(**base_fields: Any) -> Callable[..., str]:
    """返回预绑定公共字段的日志写入函数。"""

    def write_event(**event_fields: Any) -> str:
        payload = {**base_fields, **event_fields}
        return observe_agent_event(**payload)

    return write_event


def read_agent_events(
    *,
    path: Path | str = AGENT_LOG_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读取 agent 日志并返回 (valid_events, invalid_lines)。"""
    log_path = Path(path)
    if not log_path.exists():
        return [], []

    valid_events: list[dict[str, Any]] = []
    invalid_lines: list[dict[str, Any]] = []

    with log_path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            text = raw_line.strip()
            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                invalid_lines.append(
                    {
                        "line_no": line_no,
                        "reason": f"json_decode_error: {error}",
                        "raw": text,
                    }
                )
                continue

            if not isinstance(payload, dict):
                invalid_lines.append(
                    {
                        "line_no": line_no,
                        "reason": "not_json_object",
                        "raw": text,
                    }
                )
                continue

            missing_keys = sorted([key for key in _REQUIRED_EVENT_KEYS if key not in payload])
            if missing_keys:
                invalid_lines.append(
                    {
                        "line_no": line_no,
                        "reason": "missing_required_keys:" + ",".join(missing_keys),
                        "raw": text,
                    }
                )
                continue

            valid_events.append(payload)

    return valid_events, invalid_lines


def summarize_agent_log_health(*, path: Path | str = AGENT_LOG_PATH) -> dict[str, Any]:
    """汇总日志健康度，便于快速排查坏行。"""
    valid_events, invalid_lines = read_agent_events(path=path)
    return {
        "path": str(path),
        "valid_count": len(valid_events),
        "invalid_count": len(invalid_lines),
        "invalid_preview": invalid_lines[:20],
    }
