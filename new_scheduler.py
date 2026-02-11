import asyncio
import heapq
import itertools
from typing import Any, Callable, List, Optional, Tuple

class Scheduler:
    """
    完全整合的优先级任务调度器，支持动态调整并发数。
    """

    def __init__(self, max_workers: int, maxsize: int = 0):
        if max_workers <= 0:
            raise ValueError("max_workers must be > 0")
        self.max_workers = max_workers
        self.maxsize = maxsize

        self._queue: List['_Task'] = [] # type: ignore
        self._counter = itertools.count()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

        self._workers: List[asyncio.Task] = []
        self._running = False

    class _Task:
        __slots__ = ('priority', 'data', 'order', 'future')
        def __init__(self, priority: int, data: Any, future: asyncio.Future, order: int):
            if not 0 <= priority <= 15:
                raise ValueError("priority must be 0-15")
            self.priority = priority
            self.data = data
            self.future = future
            self.order = order

        def __lt__(self, other):
            if self.priority != other.priority:
                return self.priority < other.priority
            return self.order < other.order

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    async def start(self):
        """启动所有 worker 协程"""
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]

    async def stop(self, wait_for_pending: bool = True, drop_pending: bool = False) -> List[Tuple[int, Callable, tuple, dict]]:
        """
        停止所有 worker。
        :param wait_for_pending: 是否等待当前已取出的任务完成
        :param drop_pending:     是否丢弃队列中尚未执行的任务（若为False，返回这些任务的描述信息）
        :return:                 drop_pending=False时返回未执行的任务列表
        """
        self._running = False
        # 唤醒所有可能阻塞的 worker
        async with self._not_empty:
            self._not_empty.notify_all()

        # 处理队列中剩余的任务
        pending_tasks = []
        if not drop_pending:
            # 收集所有未出队的任务信息
            async with self._lock:
                while self._queue:
                    task = heapq.heappop(self._queue)
                    pending_tasks.append(
                        (task.priority, task.data[0], task.data[1], task.data[2])
                    )
                    task.future.cancel()  # 取消等待中的 Future
        else:
            # 直接取消所有队列中的 Future
            async with self._lock:
                while self._queue:
                    task = heapq.heappop(self._queue)
                    task.future.cancel()

        if not wait_for_pending:
            for w in self._workers:
                w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        return pending_tasks

    async def resize(self, max_workers: int):
        """
        动态调整并发 worker 数量。
        :param max_workers: 新的并发数（必须 > 0）
        """
        if max_workers <= 0:
            raise ValueError("max_workers must be > 0")
        old_count = len(self._workers)
        if max_workers == old_count:
            return
        elif max_workers > old_count:
            # 增加 worker
            for i in range(old_count, max_workers):
                self._workers.append(
                    asyncio.create_task(self._worker(f"worker-{i}"))
                )
        else:
            # 减少 worker：取消多余的 worker
            to_remove = old_count - max_workers
            # 选择最后 to_remove 个 worker 取消（也可随机）
            for w in self._workers[-to_remove:]:
                w.cancel()
            # 等待这些 worker 退出并移除
            done, pending = await asyncio.wait(
                self._workers[-to_remove:],
                return_when=asyncio.ALL_COMPLETED
            )
            self._workers = self._workers[:-to_remove]
        self.max_workers = max_workers

    async def add_task(self, priority: int, func: Callable, *args, **kwargs) -> Any:
        """
        提交一个任务，返回最终执行结果。
        若队列已满，抛出 asyncio.QueueFull。
        """
        future = asyncio.get_running_loop().create_future()
        order = next(self._counter)
        task = self._Task(
            priority=priority,
            data=(func, args, kwargs),
            future=future,
            order=order
        )
        async with self._not_empty:
            if self.maxsize > 0 and len(self._queue) >= self.maxsize:
                raise asyncio.QueueFull()
            heapq.heappush(self._queue, task)
            self._not_empty.notify()
        return await future

    # ------------------------------------------------------------------
    # 内部 worker 逻辑
    # ------------------------------------------------------------------
    async def _worker(self, name: str):
        while self._running:
            task = await self._pop_task()
            if task is None:
                break
            func, args, kwargs = task.data
            future = task.future
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                if not future.done():
                    future.set_result(result)
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                break
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    async def _pop_task(self) -> Optional['_Task']:
        async with self._not_empty:
            while self._running and not self._queue:
                await self._not_empty.wait()
            if not self._running:
                return None
            return heapq.heappop(self._queue)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def qsize(self) -> int:
        return len(self._queue)

    def empty(self) -> bool:
        return len(self._queue) == 0

    def full(self) -> bool:
        return self.maxsize > 0 and len(self._queue) >= self.maxsize
    

scheduler=Scheduler(5,100)