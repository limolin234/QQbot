from ncatbot.core import BotClient
from ...scheduler.registry import action_registry
import logging

logger = logging.getLogger(__name__)

async def send_group_msg(group_id: str, message: str) -> None:
    """Action: Send a message to a group."""
    if not group_id or not message:
        logger.warning("[Scheduler] send_group_msg called with empty group_id or message")
        return
    
    try:
        from bot import bot
        # Ensure group_id is string
        await bot.api.post_group_msg(str(group_id), text=str(message))
        logger.info(f"[Scheduler] Sent group message to {group_id}")
    except Exception as e:
        logger.error(f"[Scheduler] Error sending group message to {group_id}: {e}")

async def send_private_msg(user_id: str, message: str) -> None:
    """Action: Send a private message to a user."""
    if not user_id or not message:
        logger.warning("[Scheduler] send_private_msg called with empty user_id or message")
        return
        
    try:
        from bot import bot
        # Ensure user_id is string
        await bot.api.post_private_msg(str(user_id), text=str(message))
        logger.info(f"[Scheduler] Sent private message to {user_id}")
    except Exception as e:
        logger.error(f"[Scheduler] Error sending private message to {user_id}: {e}")

# Register actions
action_registry.register("core.send_group_msg", send_group_msg)
action_registry.register("core.send_private_msg", send_private_msg)
