"""工作流 Agent 配置加载工具。"""

from __future__ import annotations

import json
import os
from typing import Any,Dict,Tuple

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

    # 优先查找当前目录，若无则查找上级目录（适配子目录中的 agent）
    # 同时，如果 module_file 是虚拟文件名（不含路径），我们假设它在 workflows/agent_config.yaml 中
    
    search_dirs = [module_dir, os.path.dirname(module_dir)]
    # Special handling for scheduler config or other top-level configs if module_file is just "scheduler_manager.py"
    # But usually module_file passed here is __file__ (absolute path).
    
    # If the file is in workflows/scheduler/, its parent is workflows/scheduler, grandparent is workflows.
    # We might need to search grandparent too if the config is in workflows/agent_config.yaml
    search_dirs.append(os.path.dirname(os.path.dirname(module_dir)))

    for directory in search_dirs:
        # 1. Try YAML
        yaml_path = os.path.join(directory, "agent_config.yaml")
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

        # 2. Try JSON
        json_path = os.path.join(directory, "agent_config.json")
        if os.path.exists(json_path):
            config = load_agent_config_by_filename(current_file_name, config_path=json_path)
            if config:
                return config
    
    # Debug info (only visible in dev)
    # print(f"Failed to load config for {current_file_name} in {search_dirs}")
    return {}


_cache: Dict[Tuple[str, str], bool] = {}

def check_config(segment_name: str, config_dir: str) -> bool:
    """
    判断指定目录中配置文件是否包含 segment_name 配置段，带缓存提高性能。
    """
    global _cache
    cache_key = (config_dir, segment_name)
    if cache_key in _cache:
        return _cache[cache_key]

    yaml_path = os.path.join(config_dir, "agent_config.yaml")
    if yaml is not None and os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
            if isinstance(payload, dict):
                segment = payload.get(segment_name)
                if (
                    isinstance(segment, dict)
                    and "config" in segment
                    and isinstance(segment["config"], dict)
                    and segment["config"]
                ):
                    _cache[cache_key] = True
                    return True
        except (OSError, yaml.YAMLError):
            pass

    json_path = os.path.join(config_dir, "agent_config.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                segment = payload.get(segment_name)
                if (
                    isinstance(segment, dict)
                    and "config" in segment
                    and isinstance(segment["config"], dict)
                    and segment["config"]
                ):
                    _cache[cache_key] = True
                    return True
        except (OSError, json.JSONDecodeError):
            pass

    _cache[cache_key] = False
    return False


def get_model_name(
    config: dict[str, Any],
    stage: str | None = None,
    rule: dict[str, Any] | None = None,
) -> str:
    """
    Resolve model name based on priority:
    1. rule.decision_model (if stage='decision')
    2. config.decision_model (if stage='decision')
    3. rule.model
    4. config.model
    
    If stage != 'decision', we always fall back to 'model' (unless rule.model overrides).
    We do NOT support 'process_model' or 'reply_model' anymore.
    """
    # Base fallback
    default_model = str(config.get("model") or "").strip()
    
    # 1. Rule Level (Decision Override)
    if stage == "decision" and rule and isinstance(rule, dict):
        rule_decision_model = str(rule.get("decision_model") or "").strip()
        if rule_decision_model:
            return rule_decision_model

    # 2. Config Level (Decision Override)
    if stage == "decision":
        config_decision_model = str(config.get("decision_model") or "").strip()
        if config_decision_model:
            return config_decision_model

    # 3. Rule Level (Main Model)
    if rule and isinstance(rule, dict):
        rule_model = str(rule.get("model") or "").strip()
        if rule_model:
            return rule_model

    # 4. Config Level (Main Model)
    if default_model:
        return default_model
    
    # Error Handling for Missing Main Model
    if stage != "decision":
        # If we need a main model (not decision) and none is found, we should warn or error.
        # However, to keep backward compatibility and robustness, we fallback to a hardcoded default
        # but print a warning to the console.
        print(f"[WARNING] Agent config missing 'model' field. Falling back to 'gpt-4o-mini' for stage='{stage}'.")

    return "gpt-4o-mini"  # Ultimate fallback

