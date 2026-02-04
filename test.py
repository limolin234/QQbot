import asyncio
from ncatbot.core import BotClient,PrivateMessage,GroupMessage

bot=BotClient()

@bot.group_event()
async def on_group_message(msg:GroupMessage):
    massage_hander()

bot.run()