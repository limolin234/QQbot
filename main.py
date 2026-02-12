import asyncio,aiocron
from ncatbot.core import PrivateMessage, GroupMessage
from agent_pool import setup_agent_pool
from bot import bot, QQnumber
from workflows.auto_reply import auto_reply_pending_worker, enqueue_auto_reply_if_monitored
from workflows.forward import enqueue_forward_by_monitor_group
from workflows.summary import daily_summary, process_group_message, process_private_message

@bot.private_event()# type: ignore
async def on_private_message(msg: PrivateMessage):
    await enqueue_auto_reply_if_monitored(msg, chat_type="private")
    await process_private_message(msg)
    if msg.raw_message == "测试":
        await bot.api.post_private_msg(msg.user_id, text="NcatBot测试成功")
    elif msg.user_id == QQnumber and msg.raw_message.strip() == "/summary":
        await bot.api.post_private_msg(msg.user_id, text="收到 /summary，正在执行一次手动总结…")
        await daily_summary(run_mode="manual")
        await bot.api.post_private_msg(msg.user_id, text="手动总结任务已投递到队列，请稍等结果私聊消息。")

@bot.group_event()# type: ignore
async def on_group_message(msg: GroupMessage):
    await enqueue_auto_reply_if_monitored(msg, chat_type="group")
    await process_group_message(msg)
    await enqueue_forward_by_monitor_group(msg)
    
@bot.startup_event()# type: ignore
async def on_startup(*args):
    await setup_agent_pool()
    asyncio.create_task(auto_reply_pending_worker())
    aiocron.crontab('0 22 * * *', func=lambda: daily_summary(run_mode="auto"))

bot.run()
