from ncatbot.core import BotClient, PrivateMessage, GroupMessage
import re
from scheduler import PriorityTask, TaskType
from bot import bot,QQnumber

async def handle_task(task: PriorityTask):
    if task.type == TaskType.SUMMARY:
        summary = 'summary'
        await bot.api.post_private_msg(QQnumber, text = summary)
        print(summary)
    elif task.type == TaskType.FROWARD:
        await bot.api.post_private_msg(QQnumber, text = task.msg.raw_message)
        print(task.msg.raw_message)
    elif task.type == TaskType.GROUPNOTE:
        print("回复")
    else:
        return