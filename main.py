from ncatbot.core import PrivateMessage, GroupMessage
from agent_pool import setup_agent_pool
from workflows.agent_config_loader import check_config
from bot import bot
from workflows import message_observe, agent_observe
from workflows import summary, auto_reply, forward, badminton_monitor, safety_checkin
from workflows.router import route_private
from workflows import safety_checkin_scheduler as _safety_scheduler


def _get_admin_qq() -> str:
    return str(_safety_scheduler.SAFETY_CONFIG.get("admin_qq") or "").strip()


@bot.private_event()  # type: ignore
async def on_private_message(msg: PrivateMessage):
    await message_observe.private_entrance(msg)
    if check_config("summary_config", "./workflows"):
        await summary.private_entrance(msg)

    await route_private(
        msg,
        admin_qq=_get_admin_qq(),
        volunteer_monitor_enabled=check_config("volunteer_monitor_config", "./workflows"),
        badminton_monitor_enabled=check_config("badminton_monitor_scheduler_config", "./workflows"),
        dida_enabled=check_config("dida_agent_config", "./workflows"),
        reminder_enabled=check_config("reminder_config", "./workflows"),
        safety_checkin_enabled=check_config("safety_checkin_config", "./workflows"),
        auto_reply_fn=lambda m: auto_reply.entrance(m, chat_type="private"),
        volunteer_monitor_fn=lambda m: None,
        badminton_monitor_fn=badminton_monitor.private_entrance,
        dida_fn=lambda m: None,
        reminder_fn=lambda m: None,
        safety_checkin_fn=safety_checkin.private_entrance,
    )


@bot.group_event()  # type: ignore
async def on_group_message(msg: GroupMessage):
    await message_observe.group_entrance(msg)
    if check_config("forward_config", "./workflows"):
        await forward.group_entrance(msg)
    if check_config("auto_reply_config", "./workflows"):
        await auto_reply.entrance(msg, chat_type="group")
    if check_config("badminton_monitor_scheduler_config", "./workflows"):
        await badminton_monitor.group_entrance(msg)
    if check_config("safety_checkin_config", "./workflows"):
        await safety_checkin.group_entrance(msg)


@bot.startup_event()  # type: ignore
async def on_startup(*args):
    await setup_agent_pool()
    message_observe.start_up()
    agent_observe.start_up()
    if check_config("summary_config", "./workflows"):
        await summary.start_up()
    if check_config("auto_reply_config", "./workflows"):
        await auto_reply.start_up()
    if check_config("badminton_monitor_scheduler_config", "./workflows"):
        badminton_monitor.start_up()
    if check_config("safety_checkin_config", "./workflows"):
        safety_checkin.start_up()

bot.run()
