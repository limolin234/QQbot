import asyncio
import logging
from typing import Callable, Awaitable, Any, Dict, List

ActionFunc = Callable[..., Awaitable[Any]]
logger = logging.getLogger(__name__)

class ActionRegistry:
    def __init__(self):
        self._actions: Dict[str, ActionFunc] = {}

    def register(self, action_id: str, func: ActionFunc) -> None:
        """Register a new action."""
        if action_id in self._actions:
            logger.warning(f"[Scheduler] Overwriting action '{action_id}'")
        self._actions[action_id] = func
        logger.debug(f"[Scheduler] Registered action: {action_id}")

    def get(self, action_id: str) -> ActionFunc:
        """Retrieve an action by ID."""
        if action_id not in self._actions:
            raise ValueError(f"Action '{action_id}' not found in registry.")
        return self._actions[action_id]

    def list_actions(self) -> List[str]:
        return list(self._actions.keys())

# Global instance
action_registry = ActionRegistry()
