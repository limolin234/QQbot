import asyncio,aiocron
from datetime import datetime, timedelta
from ncatbot.core import PrivateMessage, GroupMessage
from agent_pool import setup_agent_pool
from bot import bot, QQnumber
from workflows.auto_reply import auto_reply_pending_worker, enqueue_auto_reply_if_monitored
from workflows.dida_scheduler import dida_scheduler
from workflows.forward import enqueue_forward_by_monitor_group
from workflows.summary import daily_summary, process_group_message, process_private_message

@bot.private_event()# type: ignore
async def on_private_message(msg: PrivateMessage):
    if await dida_scheduler.handle_command(msg):
        return
    await enqueue_auto_reply_if_monitored(msg, chat_type="private")
    await process_private_message(msg)
    if msg.user_id == QQnumber and msg.raw_message.strip().startswith("/summary"):
        parts = msg.raw_message.strip().split(maxsplit=1)
        target_date = None
        if len(parts) > 1:
            arg = parts[1].strip()
            if arg in ["yesterday", "昨天"]:
                target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                try:
                    datetime.strptime(arg, "%Y-%m-%d")
                    target_date = arg
                except ValueError:
                    await bot.api.post_private_msg(msg.user_id, text=f"日期格式错误：{arg}，请使用 YYYY-MM-DD 或 '昨天'")
                    return

        status_text = f"收到 /summary，正在执行一次手动总结{f' (日期: {target_date})' if target_date else ''}…"
        await bot.api.post_private_msg(msg.user_id, text=status_text)
        await daily_summary(run_mode="manual", target_date=target_date)
        await bot.api.post_private_msg(msg.user_id, text="手动总结任务已投递到队列，请稍等结果私聊消息。")

@bot.group_event()# type: ignore
async def on_group_message(msg: GroupMessage):
    if await dida_scheduler.handle_command(msg):
        return
    await enqueue_auto_reply_if_monitored(msg, chat_type="group")
    await process_group_message(msg)
    await enqueue_forward_by_monitor_group(msg)
    
@bot.startup_event()# type: ignore
async def on_startup(*args):
    await setup_agent_pool()
    asyncio.create_task(auto_reply_pending_worker())
    asyncio.create_task(dida_scheduler.start())
    aiocron.crontab('0 22 * * *', func=lambda: daily_summary(run_mode="auto"))

bot.run()
