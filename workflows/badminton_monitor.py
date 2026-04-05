"""
羽毛球群接龙监控入口

功能：
- 私聊接收主人的自然语言指令，管理"意向打球时间"（增/删/查）
- 监听指定 QQ 群的消息，用 AI 提取接龙信息
- 将接龙时间与意向时间做碰撞匹配，命中后自动接龙
"""

from __future__ import annotations

from workflows.badminton_monitor_scheduler import badminton_monitor_scheduler


async def private_entrance(msg):
    """处理私聊消息"""
    return await badminton_monitor_scheduler.handle_private_command(msg)


async def group_entrance(msg):
    """处理群消息"""
    return await badminton_monitor_scheduler.handle_group_message(msg)


def start_up():
    """启动初始化"""
    badminton_monitor_scheduler.start_up()
