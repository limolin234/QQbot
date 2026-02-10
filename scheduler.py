import asyncio,heapq,json,os,re
from collections import defaultdict
from datetime import datetime, timezone
from enum import IntEnum
from types import SimpleNamespace
from ncatbot.core import GroupMessage
from time import time
import logging
from bot import allowed_id,urgent_keywords,normal_keywords


class TaskPriority(IntEnum):
    REALTIME = 0   # 实时任务（最高优先级）
    NORMAL = 1     # 普通任务

class TaskType(IntEnum):
    SUMMARY = 0 
    FROWARD = 1 
    GROUPNOTE = 2

class PriorityTask:#带优先级的任务包装
    _counter = 0  # 全局计数器，保证同优先级FIFO

    def __init__(self, msg: GroupMessage, priority: TaskPriority,type: TaskType):
        self.msg = msg
        self.priority = priority
        self.type = type
        self.timestamp = time()
        PriorityTask._counter += 1
        self.order = PriorityTask._counter
    
    def __lt__(self, other): # 优先级数值越小越优先，同优先级按插入顺序
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.order < other.order


class MessageScheduler:#异步优先级任务调度器
    def __init__(self, maxsize: int = 0):#Args:maxsize: 队列最大容量，0 表示无限制
        self.maxsize = maxsize
        self._queue: list[PriorityTask] = []
        self._lock = asyncio.Lock()  # 保证并发安全
        self._not_empty = asyncio.Condition(self._lock)  # 非空条件变量
        self._not_full = asyncio.Condition(self._lock)   # 非满条件变量
        
    async def RTpush(self, msg: GroupMessage, type: TaskType): #添加实时任务，返回是否成功
        task = PriorityTask(msg, TaskPriority.REALTIME, type)
        try:
            await self._put(task)
        except asyncio.QueueFull:
            logging.warning(f"队列满，丢弃实时任务: {type.name}")

    async def push(self, msg: GroupMessage, type: TaskType): #添加普通任务，返回是否成功
        task = PriorityTask(msg, TaskPriority.NORMAL, type)
        try:
            await self._put(task)
        except asyncio.QueueFull:
            logging.debug(f"队列满，丢弃普通任务: {type.name}")  # 普通任务用 debug 级别
    
    async def _put(self, task: PriorityTask) -> None:
        async with self._not_empty:
            if self.maxsize > 0 and len(self._queue) >= self.maxsize:
                raise asyncio.QueueFull()
            heapq.heappush(self._queue, task)
            self._not_empty.notify()
    
    async def pop(self) -> PriorityTask:  #弹出最高优先级任务（实时任务优先于普通任务）
        async with self._not_empty:# 如果队列为空，等待
            while len(self._queue) == 0:
                await self._not_empty.wait() 
            task = heapq.heappop(self._queue)#弹出优先级最高的任务 O(log n)
            self._not_full.notify()# 通知等待的 push 操作
            return task
    
    def empty(self) -> bool:
        return len(self._queue) == 0
    
    def full(self) -> bool:
        if self.maxsize <= 0:
            return False
        return len(self._queue) >= self.maxsize
    
    async def qsize(self) -> int:
        async with self._lock:
            return len(self._queue)
    
    async def get_status(self) -> dict:#获取队列状态（用于调试）
        async with self._lock:
            rt_count = sum(1 for t in self._queue if t.priority == TaskPriority.REALTIME)
            normal_count = len(self._queue) - rt_count
            return {
                "total": len(self._queue),
                "realtime": rt_count,
                "normal": normal_count,
                "empty": self.empty(),
                "full": self.full(),
                "maxsize": self.maxsize if self.maxsize > 0 else "unlimited"
            }
            
            

def clean_message(raw_message):
    at_matches = re.findall(r'\[CQ:at,qq=(\d+)\]', raw_message)
    at_text = f"[@{','.join(at_matches)}]" if at_matches else ""
    cq_patterns = {
        r'\[CQ:image,file=[^]]+\]': '[图片]',
        r'\[CQ:reply,id=\d+\]': '[回复]',
        r'\[CQ:file,file=[^]]+\]': '[文件]',
        r'\[CQ:record,file=[^]]+\]': '[语音]',
        r'\[CQ:video,file=[^]]+\]': '[视频]',
    }
    cleaned = raw_message
    for pattern, replacement in cq_patterns.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    if at_matches:
        cleaned = re.sub(r'\[CQ:at,qq=\d+\]', at_text, cleaned)
    return cleaned.strip()


scheduler = MessageScheduler(maxsize=100)


LOG_FILE_PATH = "message.jsonl"
LOG_TEMP_FILE_PATH = "message.jsonl.processing"


_file_lock = asyncio.Lock()
async def processmsg(msg: GroupMessage):
    original_raw_message = getattr(msg, "raw_message", "") or ""
    cleaned_message = clean_message(original_raw_message)

    msg.raw_message = cleaned_message
    if msg.group_id not in allowed_id:#不在范围内
        return #直接忽略，不加入队列

    ts = _extract_message_ts(msg)
    user_name = _extract_user_name(msg)

    async with _file_lock: #写入日志
        await asyncio.to_thread(  # 在线程池中执行，避免阻塞事件循环
            _write_log,
            ts,
            msg.group_id,
            getattr(msg, "user_id", "unknown"),
            user_name,
            "group",
            cleaned_message,
        )
    if any(keyword in cleaned_message for keyword in urgent_keywords):#RT
        await scheduler.RTpush(msg,TaskType.GROUPNOTE)  #加入实时队列
    elif any(keyword in cleaned_message for keyword in normal_keywords):#IDLE
        await scheduler.push(msg,TaskType.FROWARD)
    else:
        return #不包含关键词的消息不加入队列


def _write_log(
    ts: str,
    group_id: str,
    user_id: str,
    user_name: str,
    chat_type: str,
    cleaned_message: str,
):
    """同步写入 JSONL 日志的辅助函数。"""
    payload = {
        "ts": ts,
        "group_id": str(group_id),
        "user_id": str(user_id),
        "user_name": user_name,
        "chat_type": chat_type,
        "cleaned_message": cleaned_message,
    }
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


async def daily_summary():
    """按天汇总日志并投递 SUMMARY 任务。

    处理流程：
    1) 在文件锁保护下将 `message.jsonl` 原子重命名为临时文件，避免与实时写入冲突。
    2) 读取临时文件并按 `(group_id, user_id)` 分组。
    3) 对每个分组做组内分块：单组内容超过 `chunk_size` 时继续切分，但保持同组语义。
    4) 将多个较小分组块合并为不超过 `chunk_size` 的最终块，减少后续 summary 次数。
    5) 把最终块逐条入队为 `TaskType.SUMMARY`，供 worker 消费。

    说明：
    - 当前格式为 JSONL：`ts/group_id/user_id/user_name/chat_type/cleaned_message`。
    - 兼容旧文本格式：`group_id|user_id:message` 与 `group_id:message`。
    - 每个最终 chunk 会写入 `GroupMessage.raw_message` 作为 summary 工作流输入。
    """
    file_path = LOG_FILE_PATH
    temp_path = LOG_TEMP_FILE_PATH
    chunk_size = 10000
    print(f"开始读取日志: {file_path}")
    try:
        async with _file_lock: #使用锁替换 防止遗漏信息
            try:
                await asyncio.to_thread(os.rename, file_path, temp_path)
            except FileNotFoundError:
                print("没有日志需要处理")
                return

        grouped_messages: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
        with open(temp_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                group_id, user_id, user_name, ts, message, chat_type = _parse_log_line(line)
                if chat_type != "group":
                    continue
                grouped_messages[(group_id, user_id, user_name)].append((ts, message))

        grouped_chunks: list[str] = []
        for (group_id, user_id, user_name), messages in grouped_messages.items():
            grouped_chunks.extend(
                _split_group_messages(group_id, user_id, user_name, messages, chunk_size)
            )

        final_chunks = _merge_small_chunks(grouped_chunks, chunk_size)
        for chunk in final_chunks:
            msg = SimpleNamespace(raw_message=chunk)
            await scheduler.push(msg, TaskType.SUMMARY)

        print(
            f"日志分组完成: groups={len(grouped_messages)}, "
            f"group_chunks={len(grouped_chunks)}, final_chunks={len(final_chunks)}"
        )
        await asyncio.to_thread(os.remove, temp_path)
        print("日志处理完成")
    except Exception as e:
        print(f"处理日志时出错: {e}")


def _parse_log_line(line: str) -> tuple[str, str, str, str, str, str]:
    """解析日志行，兼容 JSONL 与旧文本格式。

    返回: (group_id, user_id, user_name, ts, message, chat_type)
    """
    try:
        record = json.loads(line)
        if isinstance(record, dict):
            return (
                str(record.get("group_id", "unknown_group")),
                str(record.get("user_id", "unknown_user")),
                str(record.get("user_name", "unknown_user")),
                str(record.get("ts", "")),
                str(record.get("cleaned_message", record.get("raw_message", ""))).strip(),
                str(record.get("chat_type", "group")),
            )
    except json.JSONDecodeError:
        pass

    if ":" not in line:
        return "unknown_group", "unknown_user", "unknown_user", "", line, "group"

    prefix, message = line.split(":", 1)
    if "|" in prefix:
        group_id, user_id = prefix.split("|", 1)
        normalized_user_id = user_id.strip()
        return group_id.strip(), normalized_user_id, normalized_user_id, "", message.strip(), "group"

    return prefix.strip(), "unknown_user", "unknown_user", "", message.strip(), "group"


def _extract_message_ts(msg: GroupMessage) -> str:
    """提取消息时间并转为 ISO8601 字符串。"""
    ts_candidate = getattr(msg, "time", None)
    if ts_candidate is not None:
        try:
            ts_value = float(ts_candidate)
            return datetime.fromtimestamp(ts_value, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        except (TypeError, ValueError, OSError):
            pass
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _extract_user_name(msg: GroupMessage) -> str:
    """提取发送者昵称，优先群名片，其次 QQ 昵称。"""
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


def _format_summary_time(ts: str) -> str:
    """将 ISO 时间压缩为 summary 友好的 HH:MM，解析失败则返回空。"""
    if not ts:
        return ""
    try:
        normalized = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%H:%M")
    except ValueError:
        return ""


def _split_group_messages(
    group_id: str,
    user_id: str,
    user_name: str,
    messages: list[tuple[str, str]],
    chunk_size: int,
) -> list[str]:
    """单组分块：即便超大也保持同组。"""
    safe_name = (user_name or "unknown_user").replace("]", "）")
    header = f"[group:{group_id}][user:{user_id}][name:{safe_name}]"
    max_body_chars = max(1, chunk_size - len(header) - 1)
    chunks: list[str] = []
    current_body = ""

    def flush_current() -> None:
        nonlocal current_body
        if not current_body:
            return
        chunks.append(f"{header}\n{current_body}")
        current_body = ""

    for ts, message in messages:
        time_prefix = _format_summary_time(ts)
        normalized_message = f"[{time_prefix}] {message}" if time_prefix else message
        remaining = normalized_message
        while remaining:
            if not current_body:
                room = max_body_chars
            else:
                room = max_body_chars - len(current_body) - 1

            if room <= 0:
                flush_current()
                continue

            part = remaining[:room]
            remaining = remaining[room:]

            if current_body:
                current_body += "\n" + part
            else:
                current_body = part

            if remaining:
                flush_current()

    flush_current()
    if not chunks:
        chunks.append(header)
    return chunks


def _merge_small_chunks(grouped_chunks: list[str], chunk_size: int) -> list[str]:
    """多组小块合并：尽量打包到同一 chunk。"""
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
