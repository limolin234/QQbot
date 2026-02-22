from ncatbot.core import BotClient

bot = BotClient()

# 主人QQ，用于接收转发消息和触发管理命令
QQnumber = '2667221906'

# 允许互动的用户/群组白名单（如有需要可在此添加）
allowed_id={'2667221906','1830740938'}

urgent_keywords = {'紧急', '重要', '立刻', '马上'}
normal_keywords = {'通知', '公告'}
