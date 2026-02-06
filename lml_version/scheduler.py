import asyncio,heapq,os,re
from enum import IntEnum
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


_file_lock = asyncio.Lock()
async def processmsg(msg: GroupMessage):
    msg.raw_message=clean_message(msg.raw_message)
    if msg.group_id not in allowed_id:#不在范围内
        return #直接忽略，不加入队列
    async with _file_lock: #写入日志
        await asyncio.to_thread(  # 在线程池中执行，避免阻塞事件循环
            _write_log, msg.group_id, msg.raw_message
        )
    if any(keyword in msg.raw_message for keyword in urgent_keywords):#RT
        await scheduler.RTpush(msg,TaskType.GROUPNOTE)  #加入实时队列
    elif any(keyword in msg.raw_message for keyword in normal_keywords):#IDLE
        await scheduler.push(msg,TaskType.FROWARD)
    else:
        return #不包含关键词的消息不加入队列


def _write_log(group_id: str, message: str):
    """同步写入日志的辅助函数"""
    with open('today.log', 'a', encoding='utf-8') as f:
        f.write(f"{group_id}:{message}\n")
        f.flush()


async def daily_summary():
    file_path = 'today.log'
    temp_path = 'today.log.processing'
    print(f"开始读取日志: {file_path}")
    try:
        async with _file_lock: #使用锁替换 防止遗漏信息
            try:
                await asyncio.to_thread(os.rename, file_path, temp_path)
            except FileNotFoundError:
                print("没有日志需要处理")
                return
        with open(temp_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(10000)
                if not chunk:
                    break
                msg = GroupMessage()  # type: ignore
                msg.raw_message = chunk
                await scheduler.push(msg, TaskType.SUMMARY)
        await asyncio.to_thread(os.remove, temp_path)
        print("日志处理完成")
    except Exception as e:
        print(f"处理日志时出错: {e}")
