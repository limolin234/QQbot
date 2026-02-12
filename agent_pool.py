import asyncio
import heapq
import inspect
import logging
from time import time
from typing import Any, Callable, Coroutine, List, Optional
import uuid

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 优先级队列核心（线程安全）
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 全局状态
# ----------------------------------------------------------------------
_scheduler: Optional[PriorityScheduler] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_worker_tasks: List[asyncio.Task] = []
_worker_counter = 0

# LLM 处理器（默认为 None，启动后必须注册）
_llm_handler: Optional[Callable[[Any], Coroutine[Any, Any, Any]]] = None


# ----------------------------------------------------------------------
# 注册接口（装饰器 + 函数）
# ----------------------------------------------------------------------
def register_llm_handler(func: Optional[Callable[[Any], Coroutine]] = None):
    """
    装饰器：注册 LLM 处理器。
    用法：
        @register_llm_handler
        async def my_llm_handler(data): ...
    或：
        register_llm_handler(my_llm_handler)
    """
    def decorator(f: Callable[[Any], Coroutine]) -> Callable[[Any], Coroutine]:
        global _llm_handler
        _llm_handler = f
        logger.info("LLM 处理器已注册: %s", f.__name__)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)


# ----------------------------------------------------------------------
# Worker 池（固定数量，串行执行）
# ----------------------------------------------------------------------
async def _worker(worker_id: int):
    while True:
        task = None
        try:
            task = await _scheduler.pop() # type: ignore[union-attr]
            if task is None:
                break
            if task.data.get("type") == "__retire_worker__":
                logger.info("Worker-%d 收到缩容信号，退出", worker_id)
                break
            try:
                result = await _execute_task_payload(task.data)
                if task.future and not task.future.done():
                    task.future.set_result(result)
            except Exception as e:
                logger.exception("Worker-%d 处理任务失败, task_id=%s", worker_id, task.task_id)
                if task.future and not task.future.done():
                    task.future.set_exception(e)

        except asyncio.CancelledError:
            logger.info("Worker-%d 已取消", worker_id)
            break
        except Exception:
            logger.exception("Worker-%d 内部异常", worker_id)


def _compact_worker_tasks() -> None:
    global _worker_tasks
    _worker_tasks = [task for task in _worker_tasks if not task.done()]


def _spawn_workers(count: int) -> None:
    global _worker_counter
    for _ in range(count):
        worker_id = _worker_counter
        _worker_counter += 1
        _worker_tasks.append(asyncio.create_task(_worker(worker_id), name=f"AgentWorker-{worker_id}"))


async def _execute_task_payload(data: dict[str, Any]) -> Any:
    """
    - 根据 payload["type"] 分支：
    - callable：执行函数（submit_agent_job(...) 走这个分支），支持：
        - run_in_thread=True 时用 asyncio.to_thread 跑同步阻塞函数
        - 否则直接调用；若返回 awaitable 就 await 
    - llm：走老的 _llm_handler（给 agentp_LLM(...) 兼容保留）
    """
    payload_type = str(data.get("type", ""))
    if payload_type == "callable":
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

    if payload_type == "llm":
        if _llm_handler is None:
            raise RuntimeError("LLM 处理器未注册，请先调用 register_llm_handler")
        return await _llm_handler(data.get("payload"))

    raise ValueError(f"未知任务类型: {payload_type}")


# ----------------------------------------------------------------------
# 启动 / 停止
# ----------------------------------------------------------------------
async def setup_agent_pool(
    worker_count: int = 5,
    maxsize: int = 100,
    loop: Optional[asyncio.AbstractEventLoop] = None
) -> None:
    """启动代理池（只需要 Worker 数量和队列容量）"""
    global _scheduler, _loop, _worker_tasks, _worker_counter

    if _scheduler is not None:
        logger.warning("Agent 池已启动，忽略重复调用")
        return

    _scheduler = PriorityScheduler(maxsize=maxsize)
    _loop = loop or asyncio.get_running_loop()
    _worker_tasks = []
    _worker_counter = 0
    _spawn_workers(worker_count)
    logger.info("✅ Agent 池已启动，Worker 数量=%d，队列容量=%s",
                worker_count, maxsize if maxsize > 0 else "无限制")


async def resize_agent_pool(
    worker_count: int,
    *,
    graceful: bool = True,
    wait_timeout: float = 10.0,
) -> None:
    """动态调整 Agent 池 worker 数量。"""
    if worker_count <= 0:
        raise ValueError("worker_count must be > 0")
    if _scheduler is None:
        raise RuntimeError("❌ Agent 池未启动，请先调用 setup_agent_pool()")

    _compact_worker_tasks()
    current_count = len(_worker_tasks)
    if worker_count == current_count:
        return

    if worker_count > current_count:
        _spawn_workers(worker_count - current_count)
        logger.info("Agent 池扩容完成: %d -> %d", current_count, worker_count)
        return

    to_remove = current_count - worker_count
    if graceful:
        for _ in range(to_remove):
            retire_task = Task(
                priority=0,
                data={"type": "__retire_worker__"},
                future=None,
            )
            await _scheduler.put(retire_task)

        end_time = asyncio.get_running_loop().time() + max(wait_timeout, 0.1)
        while True:
            _compact_worker_tasks()
            if len(_worker_tasks) <= worker_count:
                break
            if asyncio.get_running_loop().time() >= end_time:
                logger.warning("Agent 池缩容等待超时，继续执行当前状态")
                break
            await asyncio.sleep(0.05)
        logger.info("Agent 池平滑缩容完成: %d -> %d", current_count, len(_worker_tasks))
        return

    cancelled_workers = _worker_tasks[-to_remove:]
    for worker in cancelled_workers:
        worker.cancel()
    await asyncio.gather(*cancelled_workers, return_exceptions=True)
    _compact_worker_tasks()
    logger.info("Agent 池强制缩容完成: %d -> %d", current_count, len(_worker_tasks))


async def stop_agent_pool(
    *,
    wait_for_pending: bool = True,
    drop_pending: bool = False,
) -> list[dict[str, Any]]:
    """停止代理池并按需返回未执行任务信息。"""
    global _worker_tasks, _scheduler, _loop, _llm_handler
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
    _worker_tasks.clear()
    _scheduler = None
    _loop = None
    _llm_handler = None
    pending_info = [
        {
            "priority": task.priority,
            "task_type": str(task.data.get("type", "")),
            "task_id": task.task_id,
        }
        for task in drained_tasks
    ]
    logger.info("🛑 Agent 池已停止，未执行任务=%d", len(pending_info))
    return pending_info


async def _submit_pool_task(
    *,
    payload: dict[str, Any],
    priority: int,
    timeout: float,
) -> Any:
    if _scheduler is None:
        raise RuntimeError("❌ Agent 池未启动，请先调用 setup_agent_pool()")

    loop = _loop or asyncio.get_running_loop()
    future = loop.create_future()
    task = Task(priority=priority, data=payload, future=future)

    try:
        await _scheduler.put(task)
    except asyncio.QueueFull as error:
        raise RuntimeError("Agent 池队列已满，请求被拒绝") from error
    except RuntimeError as error:
        raise RuntimeError("Agent 池已停止，拒绝接收新任务") from error

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
    """
    提交通用任务到 Agent 池调度执行（不改变任务内部实现方式）。
    
    实质上，就是 add_task 函数
    """
    run_blocking = not inspect.iscoroutinefunction(func) if run_in_thread is None else bool(run_in_thread)
    payload = {
        "type": "callable",
        "func": func,
        "args": args,
        "kwargs": kwargs,
        "run_in_thread": run_blocking,
    }
    return await _submit_pool_task(payload=payload, priority=priority, timeout=timeout)


# ----------------------------------------------------------------------
# 对外调用接口：await agentp_LLM(...) 直接得到回复
# ----------------------------------------------------------------------
async def agentp_LLM(
    api_key: str,
    prompt: str,
    api_base: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    priority: int = 7,
    timeout: float = 30.0,
    **kwargs
) -> str:
    """
    异步提交 LLM 请求，等待回复。
    参数会被打包成字典传递给已注册的 LLM 处理器。
    处理器必须返回字符串（模型回复）。
    """
    request_data = {
        "api_key": api_key,
        "api_base": api_base,
        "prompt": prompt,
        "model": model,
        **kwargs
    }
    result = await _submit_pool_task(
        payload={"type": "llm", "payload": request_data},
        priority=priority,
        timeout=timeout,
    )
    return str(result)


# ----------------------------------------------------------------------
# 公开接口
# ----------------------------------------------------------------------
__all__ = [
    "setup_agent_pool",
    "resize_agent_pool",
    "stop_agent_pool",
    "submit_agent_job",
    "agentp_LLM",
    "register_llm_handler",
]
