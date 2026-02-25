import asyncio,aiocron
from datetime import datetime, timedelta
from ncatbot.core import PrivateMessage, GroupMessage
from agent_pool import setup_agent_pool
from bot import bot
from workflows import summary, auto_reply, forward, dida

@bot.private_event()# type: ignore
async def on_private_message(msg: PrivateMessage):
    await summary.private_entrance(msg)
    await auto_reply.entrance(msg, chat_type="private")
    await dida.private_entrance(msg)


@bot.group_event()# type: ignore
async def on_group_message(msg: GroupMessage):
    await summary.group_entrance(msg)
    await forward.group_entrance(msg)
    await auto_reply.entrance(msg, chat_type="group")
    await dida.group_entrance(msg)
    
@bot.startup_event()# type: ignore
async def on_startup(*args):
    await setup_agent_pool()
    await summary.start_up()
    await auto_reply.start_up()
    await dida.start_up()

bot.run()
