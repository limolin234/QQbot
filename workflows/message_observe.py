import asyncio
import re, json
from datetime import datetime, timezone
from ncatbot.core import PrivateMessage, GroupMessage

LOG_FILE_PATH = "message.jsonl"
FILE_LOCK = asyncio.Lock()
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
        import os
        os.replace(tmp_path, LOG_FILE_PATH)


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
