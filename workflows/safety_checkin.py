"""
Safety Check-in 入口模块。

- group_entrance : 处理群消息，识别「1」接龙
- private_entrance: 处理管理员私聊命令（手动触发 / 查看状态）
- start_up      : 初始化调度器
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from ncatbot.core import GroupMessage, PrivateMessage

from bot import QQnumber, bot
from . import safety_checkin_scheduler as _scheduler


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _extract_user_name(msg: GroupMessage | PrivateMessage) -> str:
    sender = getattr(msg, "sender", None)

    def pick(data, *keys):
        for key in keys:
            if isinstance(data, dict):
                value = data.get(key)
            else:
                value = getattr(data, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    for source in (msg, sender):
        if not source:
            continue
        name = pick(source, "card", "group_card", "display_name",
                     "nickname", "nick", "user_name", "sender_nickname")
        if name:
            return name
    return str(getattr(msg, "user_id", "unknown"))


def _clean_message(raw: str) -> str:
    """去除 CQ 码，保留纯文本。"""
    # 去除 [CQ:at,...]
    text = re.sub(r"\[CQ:at,qq=[^\]]+\]", "", raw)
    # 去除其他 CQ 码
    text = re.sub(r"\[CQ:[^\]]+\]", "", text)
    return text.strip()


# ─────────────────────────────────────────────
# 群消息入口
# ─────────────────────────────────────────────

async def group_entrance(msg: GroupMessage) -> bool:
    """处理群消息，识别「1」接龙。"""
    group_id = str(getattr(msg, "group_id", "") or "").strip()
    user_id = str(getattr(msg, "user_id", "") or "").strip()
    raw_message = getattr(msg, "raw_message", "") or ""

    # 清理消息，只保留纯文本
    cleaned = _clean_message(raw_message)

    # 尝试识别接龙
    handled = _scheduler.handle_group_message_for_checkin(group_id, user_id, cleaned)
    return handled


# ─────────────────────────────────────────────
# 私聊入口（管理员命令）
# ─────────────────────────────────────────────

async def private_entrance(msg: PrivateMessage) -> bool:
    """处理管理员私聊命令：/safety_checkin status/start"""
    user_id = str(getattr(msg, "user_id", "") or "").strip()
    raw_text = getattr(msg, "raw_message", "") or ""
    content = _clean_message(raw_text)

    # 鉴权：必须是 admin_qq
    cfg = _scheduler.SAFETY_CONFIG
    admin_qq = str(cfg.get("admin_qq") or QQnumber or "").strip()
    if user_id != admin_qq:
        return False

    content_lower = content.lower()

    if content_lower.startswith("/safety_checkin"):
        await _handle_command(msg, content)
        return True

    return False


async def _handle_command(msg: PrivateMessage, content: str) -> None:
    """处理具体命令。"""
    user_id = str(getattr(msg, "user_id", "") or "").strip()
    parts = content.strip().split()
    cmd = parts[1].lower() if len(parts) > 1 else ""

    if cmd == "status":
        await _cmd_status(user_id)
    elif cmd == "start":
        await _cmd_start(user_id, test_mode=False)
    elif cmd == "test":
        # /safety_checkin test [轮数]，默认6轮
        rounds = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 6
        await _cmd_start(user_id, test_mode=True)
    else:
        await bot.api.post_private_msg(
            user_id,
            text="报平安管理命令：\n"
                 "/safety_checkin status  - 查看当前状态\n"
                 "/safety_checkin start   - 手动开始一轮报平安\n"
                 "/safety_checkin test   - 测试模式（3秒间隔，完整3轮）\n"
                 "/safety_checkin test 3 - 测试指定轮数\n"
                 "\n— [报平安]"
        )


async def _cmd_status(user_id: str) -> None:
    """查看当前报平安状态。"""
    cfg = _scheduler.SAFETY_CONFIG
    groups = cfg.get("monitor_groups") or []
    members = cfg.get("members") or {}
    group_id = groups[0] if groups else "未配置"

    done, total = _scheduler._get_checked_count(str(group_id))
    pending = _scheduler._get_pending_members(str(group_id))
    pending_names = [name for _, name in pending]

    text = (
        f"【报平安状态】\n"
        f"监控群：{group_id}\n"
        f"应报到：{len(members)} 人\n"
        f"已报到：{done} 人\n"
        f"未报到（{len(pending_names)} 人）：{', '.join(pending_names) or '无'}\n"
        f"\n— [报平安]"
    )
    await bot.api.post_private_msg(user_id, text=text)


async def _cmd_start(user_id: str, test_mode: bool = False) -> None:
    """手动触发报平安，test_mode=True 时间隔缩短为 3 秒。"""
    mode_str = "测试模式" if test_mode else "正式"
    _scheduler._log(f"Manual trigger ({mode_str}) from admin command.")
    asyncio.create_task(
        _scheduler.safety_checkin_scheduler._run_cycle(test_mode=test_mode)
    )
    await bot.api.post_private_msg(
        user_id,
        text=f"报平安流程已以{mode_str}触发，请查看群消息。\n"
             f"（测试模式每轮间隔 3 秒，正式模式请配置 interval_minutes）\n\n— [报平安]"
    )


# ─────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────

def start_up() -> None:
    """初始化调度器，启动定时任务。"""
    _scheduler.safety_checkin_scheduler.start_up()
