import asyncio,heapq,json,re
from datetime import datetime, timezone
from enum import IntEnum
from types import SimpleNamespace
from ncatbot.core import GroupMessage, PrivateMessage
from time import time
import logging
from bot import allowed_id,urgent_keywords
from workflows.summary import build_summary_chunks_from_log_lines
from workflows.forward import get_forward_monitor_group_ids
from workflows.auto_reply import get_auto_reply_monitor_numbers


class TaskPriority(IntEnum):
    REALTIME = 0   # 实时任务（最高优先级）
    NORMAL = 1     # 普通任务

class TaskType(IntEnum):
    SUMMARY = 0 
    FROWARD = 1 
    GROUPNOTE = 2
    AUTO_REPLY = 3

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
_file_lock = asyncio.Lock()
async def processmsg(msg: GroupMessage):
    original_raw_message = getattr(msg, "raw_message", "") or ""
    cleaned_message = clean_message(original_raw_message)

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

    msg.raw_message = cleaned_message

    if msg.group_id not in allowed_id:#不在范围内
        return #直接忽略，不加入队列
    if any(keyword in cleaned_message for keyword in urgent_keywords):#RT
        await scheduler.RTpush(msg,TaskType.GROUPNOTE)  #加入实时队列
    else:
        return #不包含关键词的消息不加入队列



async def process_private_msg(msg: PrivateMessage):
    """记录私聊消息到 message.jsonl。"""
    original_raw_message = getattr(msg, "raw_message", "") or ""
    cleaned_message = clean_message(original_raw_message)
    ts = _extract_message_ts(msg)
    user_name = _extract_user_name(msg)

    async with _file_lock:
        await asyncio.to_thread(
            _write_log,
            ts,
            "private",
            getattr(msg, "user_id", "unknown"),
            user_name,
            "private",
            cleaned_message,
        )

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


async def daily_summary(run_mode: str = "manual"):
    """按天汇总日志并投递 SUMMARY 任务。

    run_mode:
    - manual: 手动 /summary，按时间游标增量处理，不删除日志。
    - auto:   定时任务，处理全量日志，不使用手动游标。

    说明：
    - 当前格式为 JSONL：`ts/group_id/user_id/user_name/chat_type/cleaned_message`。
    - 兼容旧文本格式：`group_id|user_id:message` 与 `group_id:message`。
    - 每个最终 chunk 会写入 `GroupMessage.raw_message` 作为 summary 工作流输入。
    """
    file_path = LOG_FILE_PATH
    chunk_size = 10000
    print(f"开始读取日志: {file_path}")
    try:
        async with _file_lock:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_lines = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                print("没有日志需要处理")
                return

        final_chunks, meta = build_summary_chunks_from_log_lines(
            raw_lines,
            chunk_size=chunk_size,
            run_mode=run_mode,
        )
        if not final_chunks:
            print(
                "没有需要汇总的消息: "
                f"scope={meta.get('scope', 'group')}, "
                f"mode={meta.get('run_mode', run_mode)}, "
                f"cursor={meta.get('cursor_before', '(none)')}"
            )
            return

        for chunk in final_chunks:
            msg = SimpleNamespace(raw_message=chunk)
            await scheduler.RTpush(msg, TaskType.SUMMARY) # 已改成RTpush

        print(
            f"日志分组完成: groups={meta.get('group_count', '0')}, "
            f"group_chunks={meta.get('group_chunks', '0')}, final_chunks={len(final_chunks)}, "
            f"scope={meta.get('scope', 'group')}, mode={meta.get('run_mode', run_mode)}, "
            f"cursor={meta.get('cursor_after', meta.get('cursor_before', '(none)'))}"
        )
        print("日志处理完成")
    except Exception as e:
        print(f"处理日志时出错: {e}")

        
def _extract_message_ts(msg: GroupMessage | PrivateMessage) -> str:
    """提取消息时间并转为 ISO8601 字符串。"""
    ts_candidate = getattr(msg, "time", None)
    if ts_candidate is not None:
        try:
            ts_value = float(ts_candidate)
            return datetime.fromtimestamp(ts_value, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        except (TypeError, ValueError, OSError):
            pass
    return datetime.now().astimezone().isoformat(timespec="seconds")


async def enqueue_forward_by_monitor_group(msg: GroupMessage) -> bool:
    """按 forward 配置中的监控群号投递转发任务。"""
    monitor_group_ids = get_forward_monitor_group_ids()
    if str(getattr(msg, "group_id", "")) not in monitor_group_ids:
        return False

    cleaned_message = clean_message(getattr(msg, "raw_message", "") or "")
    ts = _extract_message_ts(msg)
    user_name = _extract_user_name(msg)
    forward_msg = SimpleNamespace(
        raw_message=cleaned_message,
        cleaned_message=cleaned_message,
        group_id=str(getattr(msg, "group_id", "")),
        user_id=str(getattr(msg, "user_id", "unknown")),
        user_name=user_name,
        ts=ts,
    )
    await scheduler.push(forward_msg, TaskType.FROWARD)
    return True


async def enqueue_auto_reply_if_monitored(
    msg: GroupMessage | PrivateMessage,
    chat_type: str,
) -> bool:
    """按 auto_reply 监控配置投递任务，支持 group/private。"""
    target_chat_type = str(chat_type).strip().lower()
    if target_chat_type not in {"group", "private"}:
        return False

    monitor_numbers = get_auto_reply_monitor_numbers(chat_type=target_chat_type)
    monitor_value = (
        str(getattr(msg, "group_id", ""))
        if target_chat_type == "group"
        else str(getattr(msg, "user_id", ""))
    )
    if monitor_value not in monitor_numbers:
        return False

    original_raw_message = getattr(msg, "raw_message", "") or ""
    cleaned_message = clean_message(original_raw_message)
    auto_reply_msg = SimpleNamespace(
        raw_message=original_raw_message,
        cleaned_message=cleaned_message,
        chat_type=target_chat_type,
        group_id=monitor_value if target_chat_type == "group" else "private",
        user_id=str(getattr(msg, "user_id", "unknown")),
        user_name=_extract_user_name(msg),
        ts=_extract_message_ts(msg),
    )
    await scheduler.push(auto_reply_msg, TaskType.AUTO_REPLY)
    return True


def _extract_user_name(msg: GroupMessage | PrivateMessage) -> str:
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
