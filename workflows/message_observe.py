import asyncio
import re, json
import threading
import shutil
import os
from datetime import datetime, timezone
from ncatbot.core import PrivateMessage, GroupMessage

LOG_FILE_PATH = "message.jsonl"
FILE_LOCK = asyncio.Lock()
# 添加物理文件锁，用于多线程环境下的文件访问互斥（解决 Windows 文件占用问题）
PHYSICAL_FILE_LOCK = threading.Lock()
MAX_LOG_LINES = 3000
TRUNCATE_TO_LINES = 2000
_line_counter = 0


def start_up() -> None:
    global _line_counter
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            _line_counter = sum(1 for _ in f)
    except FileNotFoundError:
        _line_counter = 0


def _truncate_log() -> None:
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return

    if len(lines) > TRUNCATE_TO_LINES:
        lines = lines[-TRUNCATE_TO_LINES:]
        tmp_path = LOG_FILE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        # 使用锁保护重命名操作
        with PHYSICAL_FILE_LOCK:
            try:
                # Docker bind mount 场景下不能使用 os.replace 覆盖原文件
                # 必须使用 copy + truncate 的方式保留原文件 inode
                shutil.copyfile(tmp_path, LOG_FILE_PATH)
                os.remove(tmp_path)
            except OSError:
                pass  # 如果失败，下次再试，避免崩溃


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


def _write_log(
    ts: str,
    group_id: str,
    user_id: str,
    user_name: str,
    chat_type: str,
    cleaned_message: str,
) -> None:
    global _line_counter
    
    payload = {
        "ts": ts,
        "group_id": str(group_id),
        "user_id": str(user_id),
        "user_name": user_name,
        "chat_type": chat_type,
        "cleaned_message": cleaned_message,
    }

    with PHYSICAL_FILE_LOCK:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()

    _line_counter += 1
    if _line_counter >= MAX_LOG_LINES:
        _truncate_log()
        _line_counter = TRUNCATE_TO_LINES


async def group_entrance(msg: GroupMessage) -> None:
    cleaned_message = clean_message(getattr(msg, "raw_message", "") or "")
    ts = _extract_message_ts(msg)
    user_name = _extract_user_name(msg)

    async with FILE_LOCK:
        await asyncio.to_thread(
            _write_log,
            ts,
            str(getattr(msg, "group_id", "")),
            str(getattr(msg, "user_id", "unknown")),
            user_name,
            "group",
            cleaned_message,
        )


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
    log_path: str = LOG_FILE_PATH,
) -> list[str]:
    """从 message.jsonl 读取最近上下文消息（按 chat_type + 会话范围）。"""
    try:
        with PHYSICAL_FILE_LOCK:
            with open(log_path, "r", encoding="utf-8") as file:
                raw_lines = [line.strip() for line in file if line.strip()]
    except OSError:
        return []

    matched_records = []
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

    context_lines = []
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


async def private_entrance(msg: PrivateMessage) -> None:
    cleaned_message = clean_message(getattr(msg, "raw_message", "") or "")
    ts = _extract_message_ts(msg)
    user_name = _extract_user_name(msg)

    async with FILE_LOCK:
        await asyncio.to_thread(
            _write_log,
            ts,
            "private",
            str(getattr(msg, "user_id", "unknown")),
            user_name,
            "private",
            cleaned_message,
        )
