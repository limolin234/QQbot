"""工作流 Agent 配置加载工具。"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_agent_config_by_filename(
    file_name: str,
    *,
    config_path: str,
) -> dict[str, Any]:
    """按 `file_name` 从 JSON 配置中查找并返回 `config` 字段。"""
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
    """根据当前模块文件路径读取配置：优先 YAML，回退 JSON。"""
    module_dir = os.path.dirname(module_file)
    current_file_name = os.path.basename(module_file)

    yaml_path = os.path.join(module_dir, "agent_config.yaml")
    if yaml is not None and os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as file:
                payload = yaml.safe_load(file)
        except (OSError, yaml.YAMLError):
            payload = None

        if isinstance(payload, dict):
            for item in payload.values():
                if not isinstance(item, dict):
                    continue
                if str(item.get("file_name", "")).strip() != current_file_name:
                    continue
                config = item.get("config")
                if isinstance(config, dict):
                    return dict(config)

    config_path = os.path.join(module_dir, "agent_config.json")
    return load_agent_config_by_filename(current_file_name, config_path=config_path)
