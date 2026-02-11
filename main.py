import asyncio,aiocron
from ncatbot.core import PrivateMessage, GroupMessage
import scheduler,handler
from agent_pool import setup_agent_pool
from bot import bot, QQnumber

async def worker():
    while True:
        task = await scheduler.scheduler.pop()
        await handler.handle_task(task)

@bot.private_event()# type: ignore
async def on_private_message(msg: PrivateMessage):
    await scheduler.enqueue_auto_reply_if_monitored(msg, chat_type="private")
    await scheduler.process_private_msg(msg)
    if msg.raw_message == "测试":
        await bot.api.post_private_msg(msg.user_id, text="NcatBot测试成功")
    elif msg.user_id == QQnumber and msg.raw_message.strip() == "/summary":
        await bot.api.post_private_msg(msg.user_id, text="收到 /summary，正在执行一次手动总结…")
        await scheduler.daily_summary(run_mode="manual")
        await bot.api.post_private_msg(msg.user_id, text="手动总结任务已投递到队列，请稍等结果私聊消息。")

@bot.group_event()# type: ignore
async def on_group_message(msg: GroupMessage):
    await scheduler.enqueue_auto_reply_if_monitored(msg, chat_type="group")
    await scheduler.processmsg(msg)
    await scheduler.enqueue_forward_by_monitor_group(msg)
    
@bot.startup_event()# type: ignore
async def on_startup(*args):
    await setup_agent_pool()
    asyncio.create_task(worker())
    asyncio.create_task(scheduler.auto_reply_pending_worker())
    aiocron.crontab('0 22 * * *', func=lambda: scheduler.daily_summary(run_mode="auto"))

bot.run()
