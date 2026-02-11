import new_scheduler
from new_scheduler import scheduler
import asyncio

async def sample_task(name: str, delay: float = 1):
    await asyncio.sleep(delay)
    return f"Task {name} done"

async def main():

    await scheduler.start()

    # 1. 提交一批任务，立即创建 Task 后台运行
    tasks = [
        asyncio.create_task(scheduler.add_task(i % 3, sample_task, f"T{i}", 0.5))
        for i in range(10)
    ]

    # 2. 动态扩容到 4 个 worker
    await scheduler.resize(4)
    print("扩容到4 workers")

    # 3. 再提交几个任务，同样创建 Task
    more_tasks = [
        asyncio.create_task(scheduler.add_task(1, sample_task, f"Extra{j}", 0.3))
        for j in range(3)
    ]
    tasks.extend(more_tasks)

    # 4. 动态缩容到 2 个 worker
    await scheduler.resize(2)
    print("缩容到2 workers")

    # 5. 等待前5个任务完成，获取结果
    done, pending = await asyncio.wait(tasks[:5], return_when=asyncio.ALL_COMPLETED)
    results = [t.result() for t in done]
    print("部分结果:", results)

    # 6. 停止调度器，并返回未执行的任务信息
    pending_tasks_info = await scheduler.stop(wait_for_pending=True, drop_pending=False)
    print(f"未执行的 {len(pending_tasks_info)} 个任务: {[p[0] for p in pending_tasks_info]}")

    # 7. 清理：取消所有尚未完成的 Task，避免警告
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(main())