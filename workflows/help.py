from ncatbot.core import PrivateMessage, GroupMessage
from bot import bot

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