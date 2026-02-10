import asyncio,aiocron
from ncatbot.core import PrivateMessage, GroupMessage
import scheduler,handler
from bot import bot

async def worker():
    while True:
        task = await scheduler.scheduler.pop()
        await handler.handle_task(task)

@bot.private_event()# type: ignore
async def on_private_message(msg: PrivateMessage):
    if msg.raw_message == "测试":
        await bot.api.post_private_msg(msg.user_id, text="NcatBot测试成功")

@bot.group_event()# type: ignore
async def on_group_message(msg: GroupMessage):
    await scheduler.processmsg(msg)
    
@bot.startup_event()# type: ignore
async def on_startup(*args):
    asyncio.create_task(worker())
    aiocron.crontab('0 22 * * *', func=scheduler.daily_summary)

bot.run()