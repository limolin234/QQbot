"""Forward 工作流占位模块：当前仅负责读取 agent 配置。"""

from __future__ import annotations

from typing import Any

from .agent_config_loader import load_current_agent_config


FORWARD_AGENT_CONFIG = load_current_agent_config(__file__)


def get_forward_agent_config() -> dict[str, Any]:
    """返回 forward agent 的当前配置（来自 `workflows/agent_config.json`）。"""
    return dict(FORWARD_AGENT_CONFIG)


