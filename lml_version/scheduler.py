import asyncio,heapq
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
    
    async def RTpush(self, msg: GroupMessage,type: TaskType) -> None:#添加实时任务（最高优先级）
        task = PriorityTask(msg, TaskPriority.REALTIME,type)
        await self._put(task)
    
    async def push(self, msg: GroupMessage,type: TaskType) -> None:#添加普通任务
        task = PriorityTask(msg, TaskPriority.NORMAL,type)
        await self._put(task)
    
    async def _put(self, task: PriorityTask) -> None:#内部方法：插入任务到优先队列
        async with self._not_full:# 如果队列已满，等待
            while self.maxsize > 0 and len(self._queue) >= self.maxsize:
                await self._not_full.wait()
            heapq.heappush(self._queue, task)  # 插入任务并保持优先级顺序 O(log n)
            self._not_empty.notify() # 通知等待的 pop 操作
    
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
            
            
            
logging.basicConfig( #配置logging
    filename=f'today.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

scheduler = MessageScheduler(maxsize=100)

async def processmsg(msg: GroupMessage):
    if msg.group_id not in allowed_id:#不在范围内
        return #直接忽略，不加入队列
    elif any(keyword in msg.raw_message for keyword in urgent_keywords):#RT
        await scheduler.RTpush(msg,TaskType.GROUPNOTE)  #加入实时队列
    elif any(keyword in msg.raw_message for keyword in normal_keywords):#IDLE
        await scheduler.push(msg,TaskType.FROWARD)
    else:
        logging.info(msg)
        
async def daily_summary():
    file_path = 'today.log'
    print(f"开始读取日志: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk_number = 1
            all_summaries = []
            while True: #每次读取 10K 字符
                chunk = f.read(10000)
                if not chunk:
                    break
                msg = GroupMessage()
                msg.raw_message = chunk
                scheduler.push(msg,TaskType.SUMMARY)
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")