import asyncio
import heapq
import inspect
import logging
from time import time
from typing import Any, Callable, Optional
import uuid

logger = logging.getLogger(__name__)


class Task:
    __slots__ = ('priority', 'data', 'timestamp', 'order', 'task_id', 'future')

    def __init__(self, priority: int, data: Any, future: Optional[asyncio.Future] = None):
        if not 0 <= priority <= 15:
            raise ValueError("priority must be 0-15")
        self.priority = priority
        self.data = data
        self.timestamp = time()
        self.order = 0
        self.task_id = uuid.uuid4().hex[:8]
        self.future = future

    def __lt__(self, other: 'Task') -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.order < other.order


class PriorityScheduler:
    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self._queue: list[Task] = []
        self._counter = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._closed = False

    async def put(self, task: Task) -> None:
        async with self._not_empty:
            if self._closed:
                raise RuntimeError("scheduler is closed")
            if self.maxsize > 0 and len(self._queue) >= self.maxsize:
                raise asyncio.QueueFull()
            task.order = self._counter
            self._counter += 1
            heapq.heappush(self._queue, task)
            self._not_empty.notify()

    async def pop(self) -> Task | None:
        async with self._not_empty:
            while not self._queue:
                if self._closed:
                    return None
                await self._not_empty.wait()
            return heapq.heappop(self._queue)

    async def drain(self) -> list[Task]:
        async with self._not_empty:
            drained: list[Task] = []
            while self._queue:
                drained.append(heapq.heappop(self._queue))
            return drained

    async def close(self) -> None:
        async with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def qsize(self): return len(self._queue)
    def empty(self): return len(self._queue) == 0
    def full(self): return self.maxsize > 0 and len(self._queue) >= self.maxsize


_scheduler: Optional[PriorityScheduler] = None
_worker_tasks: set[asyncio.Task] = set()
_target_workers = 0


async def _worker(worker_id: int):
    """Worker 循环，异常时自动恢复"""
    while True:
        try:
            task = await _scheduler.pop()  # type: ignore[union-attr]
            if task is None:
                break
            
            if task.data.get("type") == "__retire_worker__":
                logger.info("Worker-%d 收到缩容信号，退出", worker_id)
                break
            
            result = await _execute_task_payload(task.data)
            if task.future and not task.future.done():
                task.future.set_result(result)
                
        except asyncio.CancelledError:
            logger.info("Worker-%d 已取消", worker_id)
            break
        except Exception:
            logger.exception("Worker-%d 处理任务失败, task_id=%s", worker_id, task.task_id if task else "unknown")


async def _execute_task_payload(data: dict[str, Any]) -> Any:
    if data.get("type") != "callable":
        raise ValueError(f"未知任务类型: {data.get('type')}")
    
    func = data.get("func")
    if not callable(func):
        raise ValueError("callable 任务缺少可调用对象")
    
    args = data.get("args", ())
    kwargs = data.get("kwargs", {})
    run_in_thread = bool(data.get("run_in_thread", False))
    
    if run_in_thread:
        return await asyncio.to_thread(func, *args, **kwargs)
    
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _spawn_workers(count: int) -> None:
    """创建指定数量的 worker"""
    for i in range(count):
        worker_id = len(_worker_tasks)
        task = asyncio.create_task(_worker(worker_id), name=f"AgentWorker-{worker_id}")
        _worker_tasks.add(task)
        task.add_done_callback(_worker_tasks.discard)


async def setup_agent_pool(
    worker_count: int = 5,
    maxsize: int = 100,
) -> None:
    """启动代理池"""
    global _scheduler, _target_workers
    
    if _scheduler is not None:
        logger.warning("Agent 池已启动，忽略重复调用")
        return
    
    _scheduler = PriorityScheduler(maxsize=maxsize)
    _target_workers = worker_count
    _spawn_workers(worker_count)
    logger.info("✅ Agent 池已启动，Worker 数量=%d，队列容量=%s",
                worker_count, maxsize if maxsize > 0 else "无限制")


async def resize_agent_pool(worker_count: int, *, graceful: bool = True, wait_timeout: float = 10.0) -> None:
    """动态调整 Worker 数量"""
    global _target_workers
    
    if worker_count <= 0:
        raise ValueError("worker_count must be > 0")
    if _scheduler is None:
        raise RuntimeError("❌ Agent 池未启动，请先调用 setup_agent_pool()")
    
    current_count = len(_worker_tasks)
    if worker_count == current_count:
        return
    
    if worker_count > current_count:
        _spawn_workers(worker_count - current_count)
        _target_workers = worker_count
        logger.info("Agent 池扩容完成: %d -> %d", current_count, worker_count)
        return
    
    # 缩容
    to_remove = current_count - worker_count
    _target_workers = worker_count
    
    if graceful:
        for _ in range(to_remove):
            retire_task = Task(priority=0, data={"type": "__retire_worker__"}, future=None)
            await _scheduler.put(retire_task)
        
        end_time = asyncio.get_running_loop().time() + max(wait_timeout, 0.1)
        while len(_worker_tasks) > worker_count:
            if asyncio.get_running_loop().time() >= end_time:
                logger.warning("Agent 池缩容等待超时，继续执行当前状态")
                break
            await asyncio.sleep(0.05)
        logger.info("Agent 池平滑缩容完成: %d -> %d", current_count, len(_worker_tasks))
    else:
        for task in list(_worker_tasks)[-to_remove:]:
            task.cancel()
        await asyncio.gather(*list(_worker_tasks)[-to_remove:], return_exceptions=True)
        logger.info("Agent 池强制缩容完成: %d -> %d", current_count, len(_worker_tasks))


async def stop_agent_pool(*, wait_for_pending: bool = True, drop_pending: bool = False) -> list[dict[str, Any]]:
    """停止代理池"""
    global _scheduler, _worker_tasks
    
    if _scheduler is None:
        return []
    
    drained_tasks: list[Task] = []
    if drop_pending or not wait_for_pending:
        drained_tasks = await _scheduler.drain()
        for task in drained_tasks:
            if task.future and not task.future.done():
                task.future.cancel()
    
    await _scheduler.close()
    
    if not wait_for_pending:
        for task in _worker_tasks:
            task.cancel()
    
    if _worker_tasks:
        await asyncio.gather(*_worker_tasks, return_exceptions=True)
    
    pending_info = [
        {"priority": t.priority, "task_type": str(t.data.get("type", "")), "task_id": t.task_id}
        for t in drained_tasks
    ]
    
    _scheduler = None
    _worker_tasks.clear()
    logger.info("🛑 Agent 池已停止，未执行任务=%d", len(pending_info))
    return pending_info


async def _submit_pool_task(*, payload: dict[str, Any], priority: int, timeout: float) -> Any:
    if _scheduler is None:
        raise RuntimeError("❌ Agent 池未启动，请先调用 setup_agent_pool()")
    
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    task = Task(priority=priority, data=payload, future=future)
    
    try:
        await _scheduler.put(task)
    except asyncio.QueueFull as e:
        raise RuntimeError("Agent 池队列已满，请求被拒绝") from e
    except RuntimeError as e:
        raise RuntimeError("Agent 池已停止，拒绝接收新任务") from e
    
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        if not future.done():
            future.cancel()
        raise


async def submit_agent_job(
    func: Callable[..., Any],
    *args: Any,
    priority: int = 7,
    timeout: float = 60.0,
    run_in_thread: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    """提交任务到 Agent 池"""
    run_blocking = not inspect.iscoroutinefunction(func) if run_in_thread is None else bool(run_in_thread)
    payload = {
        "type": "callable",
        "func": func,
        "args": args,
        "kwargs": kwargs,
        "run_in_thread": run_blocking,
    }
    return await _submit_pool_task(payload=payload, priority=priority, timeout=timeout)


__all__ = ["setup_agent_pool", "resize_agent_pool", "stop_agent_pool", "submit_agent_job"]