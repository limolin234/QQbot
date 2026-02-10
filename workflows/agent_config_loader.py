"""工作流 Agent 配置加载工具。"""

from __future__ import annotations

import json
import os
from typing import Any


def load_agent_config_by_filename(
    file_name: str,
    *,
    config_path: str,
) -> dict[str, Any]:
    """按 `file_name` 从 `agent_config.json` 中查找并返回 config 字段。"""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(payload, dict):
        return {}

    for item in payload.values():
        if not isinstance(item, dict):
            continue
        if str(item.get("file_name", "")).strip() != file_name:
            continue
        config = item.get("config")
        if isinstance(config, dict):
            return dict(config)
    return {}


def load_current_agent_config(module_file: str) -> dict[str, Any]:
    """根据当前模块文件路径，自动读取同目录 `agent_config.json` 的对应配置。"""
    module_dir = os.path.dirname(module_file)
    current_file_name = os.path.basename(module_file)
    config_path = os.path.join(module_dir, "agent_config.json")
    return load_agent_config_by_filename(current_file_name, config_path=config_path)

