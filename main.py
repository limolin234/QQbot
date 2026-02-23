import asyncio,aiocron
from datetime import datetime, timedelta
from ncatbot.core import PrivateMessage, GroupMessage
from agent_pool import setup_agent_pool
from bot import bot, QQnumber
from workflows.auto_reply import auto_reply_pending_worker, enqueue_auto_reply_if_monitored
from workflows.dida_agent import dida_agent_pending_worker, enqueue_dida_agent_if_monitored
from workflows.dida_scheduler import dida_scheduler
from workflows.forward import enqueue_forward_by_monitor_group
from workflows.summary import daily_summary, process_group_message, process_private_message

async def handle_help(msg: PrivateMessage | GroupMessage) -> bool:
    text = str(getattr(msg, "raw_message", "") or "").strip()
    if text == "/help":
        help_msg = (
            "🤖 QQBot 命令帮助\n"
            "------------------\n"
            "📌 基础命令：\n"
            "/help - 显示此帮助信息\n"
            "/dida_auth - 获取滴答清单授权链接\n"
            "/bind_dida code=xxxx - 绑定滴答清单账号\n\n"
            "🔧 管理员命令 (仅私聊)：\n"
            "/summary [date] - 手动触发日报总结 (date可选 '昨天' 或 YYYY-MM-DD)"
        )
        if isinstance(msg, GroupMessage):
             await bot.api.post_group_msg(msg.group_id, text=help_msg)
        else:
             await bot.api.post_private_msg(msg.user_id, text=help_msg)
        return True
    return False

@bot.private_event()# type: ignore
async def on_private_message(msg: PrivateMessage):
    if await handle_help(msg):
        return
    if await dida_scheduler.handle_command(msg):
        return
    await enqueue_auto_reply_if_monitored(msg, chat_type="private")
    await enqueue_dida_agent_if_monitored(msg, chat_type="private")
    await process_private_message(msg)
    if msg.user_id == QQnumber and msg.raw_message.strip() == "/summary":
        await bot.api.post_private_msg(msg.user_id, text="收到 /summary，正在执行一次手动总结…")
        await daily_summary(run_mode="manual")
        await bot.api.post_private_msg(msg.user_id, text="手动总结任务已投递到队列，请稍等结果私聊消息。")

@bot.group_event()# type: ignore
async def on_group_message(msg: GroupMessage):
    if await handle_help(msg):
        return
    if await dida_scheduler.handle_command(msg):
        return
    await enqueue_auto_reply_if_monitored(msg, chat_type="group")
    await enqueue_dida_agent_if_monitored(msg, chat_type="group")
    await process_group_message(msg)
    await enqueue_forward_by_monitor_group(msg)
    
@bot.startup_event()# type: ignore
async def on_startup(*args):
    await setup_agent_pool()
    asyncio.create_task(auto_reply_pending_worker())
    asyncio.create_task(dida_agent_pending_worker())
    asyncio.create_task(dida_scheduler.start())
    aiocron.crontab('0 22 * * *', func=lambda: daily_summary(run_mode="auto"))

bot.run()
