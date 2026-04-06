"""
私聊消息路由 Agent

职责：判断每条私聊消息属于哪个 Agent，输出路由决策，再转发给对应 Agent。
路由方式：使用 LLM 判断（AI 路由）。
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Literal

from ncatbot.core import PrivateMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from bot import bot

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ── 路由目标定义 ──────────────────────────────────────────────
RouteDest = Literal["volunteer_monitor", "badminton_monitor", "dida", "reminder", "safety_checkin", "auto_reply"]


# ── LLM 路由 ──────────────────────────────────────────────────
_ROUTER_SYSTEM = """\
你是一个消息路由判断器。判断用户的私聊消息应该路由到哪个 Agent。

可选路由目标：
- volunteer_monitor：志愿者任务监控，管理"我想工作的时间段"（添加/查看/删除志愿意向监控时间）
- badminton_monitor：羽毛球接龙监控，管理"我想打球的时间段"和自动接龙（添加/查看/删除打球意向、自动接羽毛球接龙、查看已接龙日程）
- dida：滴答清单任务管理（创建/查看/完成待办事项）
- reminder：定时提醒（设置提醒、查看提醒）
- safety_checkin：寝室报平安接龙管理（查看报平安状态、手动触发报平安流程）
- auto_reply：普通闲聊、问答、其他无法归类的消息

路由判断规则：
- 如果用户想监控志愿者/公益活动时间 → volunteer_monitor
- 如果用户想监控羽毛球接龙、设置打球意向、查看已接龙 → badminton_monitor
- 如果用户想管理待办事项 → dida
- 如果用户想设置定时提醒 → reminder
- 如果用户想查看报平安状态或触发报平安 → safety_checkin
- 其他所有消息 → auto_reply

只输出 JSON，格式：{"dest": "<路由目标>", "reason": "<一句话原因>"}
不要输出其他内容。
"""


async def _llm_route(text: str) -> tuple[RouteDest, str]:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return "auto_reply", "LLM_API_KEY 未配置，降级到 auto_reply"
    kwargs = {"model": "qwen3-max", "temperature": 0.0, "openai_api_key": api_key}
    base_url = os.getenv("LLM_API_BASE_URL")
    if base_url:
        kwargs["openai_api_base"] = base_url
    llm = ChatOpenAI(**kwargs)
    messages = [SystemMessage(content=_ROUTER_SYSTEM), HumanMessage(content=f"用户消息：{text}")]
    try:
        resp = await asyncio.to_thread(llm.invoke, messages)
        raw = str(resp.content).strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        dest = str(parsed.get("dest") or "auto_reply")
        reason = str(parsed.get("reason") or "")
        if dest not in ("volunteer_monitor", "badminton_monitor", "dida", "reminder", "safety_checkin", "auto_reply"):
            dest = "auto_reply"
        return dest, reason  # type: ignore
    except Exception as e:
        return "auto_reply", f"LLM 路由失败({e})，降级到 auto_reply"


async def _send_private(qq: str, text: str) -> None:
    try:
        await bot.api.post_private_msg(qq, text=text)
    except Exception as e:
        print(f"[ROUTER] send_private failed: {e}", flush=True)


async def route_private(
    msg: PrivateMessage,
    *,
    admin_qq: str,
    volunteer_monitor_enabled: bool,
    badminton_monitor_enabled: bool,
    dida_enabled: bool,
    reminder_enabled: bool,
    safety_checkin_enabled: bool,
    auto_reply_fn,
    volunteer_monitor_fn,
    badminton_monitor_fn,
    dida_fn,
    reminder_fn,
    safety_checkin_fn,
) -> None:
    """
    路由私聊消息到对应 Agent，并向主人发送路由决策通知。
    """
    text = str(getattr(msg, "raw_message", "") or "").strip()
    user_id = str(getattr(msg, "user_id", "") or "").strip()

    # 管理员私聊 → LLM 路由
    if user_id == admin_qq:
        dest, reason = await _llm_route(text)
        method = "AI判断"
    else:
        # 非管理员 → 默认 auto_reply
        dest = "auto_reply"
        method = "默认"
        reason = ""

    # ── 发送路由通知给主人 ──────────────────────────────────────
    if user_id == admin_qq:
        reason_display = f" · {reason}" if reason else ""
        route_notice = (
            f"[路由] {method} → {dest}{reason_display}"
        )
        await _send_private(admin_qq, route_notice)

    # ── 分发给对应 Agent ─────────────────────────────────────────
    if dest == "volunteer_monitor" and volunteer_monitor_enabled:
        await volunteer_monitor_fn(msg)
    elif dest == "badminton_monitor" and badminton_monitor_enabled:
        await badminton_monitor_fn(msg)
    elif dest == "dida" and dida_enabled:
        await dida_fn(msg)
    elif dest == "reminder" and reminder_enabled:
        await reminder_fn(msg)
    elif dest == "safety_checkin" and safety_checkin_enabled:
        await safety_checkin_fn(msg)
    else:
        await auto_reply_fn(msg)
