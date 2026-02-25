import asyncio
from ncatbot.core import PrivateMessage, GroupMessage
from .dida_agent import dida_agent_pending_worker, enqueue_dida_agent_if_monitored
from .dida_scheduler import dida_scheduler
from ..help import handle_help

async def start_up():
    asyncio.create_task(dida_agent_pending_worker())
    asyncio.create_task(dida_scheduler.start())
    
async def private_entrance(msg: PrivateMessage):
    if await handle_help(msg):
        return
    if await dida_scheduler.handle_command(msg):
        return
    await enqueue_dida_agent_if_monitored(msg, chat_type="private")
    
async def group_entrance(msg:GroupMessage):
    if await handle_help(msg):
        return
    if await dida_scheduler.handle_command(msg):
        return
    await enqueue_dida_agent_if_monitored(msg, chat_type="group")