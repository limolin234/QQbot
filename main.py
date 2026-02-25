import asyncio,aiocron,os
from datetime import datetime, timedelta
from ncatbot.core import PrivateMessage, GroupMessage
from agent_pool import setup_agent_pool
from workflows.agent_config_loader import check_config
from bot import bot
from workflows import summary, auto_reply, forward, dida

@bot.private_event()# type: ignore
async def on_private_message(msg: PrivateMessage):
    if check_config("summary_config","./workflows"):
        await summary.private_entrance(msg)
    if check_config("auto_reply_config","./workflows"):
        await auto_reply.entrance(msg, chat_type="private")
    if check_config("dida_agent_config","./workflows"):
        await dida.private_entrance(msg)


@bot.group_event()# type: ignore
async def on_group_message(msg: GroupMessage):
    if check_config("summary_config","./workflows"):
        await summary.group_entrance(msg)
    if check_config("forward_config","./workflows"):
        await forward.group_entrance(msg)
    if check_config("auto_reply_config","./workflows"):
        await auto_reply.entrance(msg, chat_type="group")
    if check_config("dida_agent_config","./workflows"):
        await dida.group_entrance(msg)
    
@bot.startup_event()# type: ignore
async def on_startup(*args):
    await setup_agent_pool()
    if check_config("summary_config","./workflows"):
        await summary.start_up()
    if check_config("auto_reply_config","./workflows"):
        await auto_reply.start_up()
    if check_config("dida_agent_config","./workflows"):
        await dida.start_up()

bot.run()
