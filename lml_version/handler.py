from ncatbot.core import BotClient, PrivateMessage, GroupMessage
from scheduler import PriorityTask, TaskType
import agent
from bot import bot

async def handle_task(task: PriorityTask):
    if task.type == TaskType.SUMMARY:
        summary = agent.summary()
        await bot.api.post_private_msg('QQnumber', text = summary)
    elif task.type == TaskType.FROWARD:
        await bot.api.post_private_msg('QQnumber', text = task.msg.raw_message)
    elif task.type == TaskType.GROUPNOTE:
        print
    else:
        return