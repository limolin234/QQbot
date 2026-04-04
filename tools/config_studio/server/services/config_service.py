from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .snapshot_service import SnapshotService


class ConfigService:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.env_path = self.repo_root / ".env"
        self.env_example_path = self.repo_root / ".env.example"
        self.agent_yaml_path = self.repo_root / "workflows" / "agent_config.yaml"
        self.agent_yaml_example_path = (
            self.repo_root / "workflows" / "agent_config.yaml.example"
        )
        self.snapshot_service = SnapshotService(repo_root)

    def load_all(self) -> tuple[dict[str, str], dict[str, Any]]:
        env_map = self.read_env()
        agent_map = self.read_agent_yaml()
        return env_map, agent_map

    def ensure_files_exist(self) -> None:
        if not self.env_path.exists() and self.env_example_path.exists():
            self.env_path.write_text(
                self.env_example_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        if not self.agent_yaml_path.exists() and self.agent_yaml_example_path.exists():
            self.agent_yaml_path.parent.mkdir(parents=True, exist_ok=True)
            self.agent_yaml_path.write_text(
                self.agent_yaml_example_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    def read_env(self) -> dict[str, str]:
        source = self.env_path if self.env_path.exists() else self.env_example_path
        if not source.exists():
            return {"LLM_API_KEY": "", "LLM_API_BASE_URL": ""}

        data: dict[str, str] = {}
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()

        return {
            "LLM_API_KEY": data.get("LLM_API_KEY", ""),
            "LLM_API_BASE_URL": data.get("LLM_API_BASE_URL", ""),
        }

    def write_env(self, llm_api_key: str, llm_api_base_url: str) -> str:
        self.ensure_files_exist()
        self.snapshot_service.create_snapshot(self.env_path)

        existing_lines = []
        if self.env_path.exists():
            existing_lines = self.env_path.read_text(encoding="utf-8").splitlines()

        replaced_key = False
        replaced_base_url = False
        new_lines: list[str] = []
        for line in existing_lines:
            stripped = line.strip()
            if stripped.startswith("LLM_API_KEY") and "=" in stripped:
                new_lines.append(f"LLM_API_KEY = {llm_api_key}")
                replaced_key = True
                continue
            if stripped.startswith("LLM_API_BASE_URL") and "=" in stripped:
                new_lines.append(f"LLM_API_BASE_URL = {llm_api_base_url}")
                replaced_base_url = True
                continue
            new_lines.append(line)

        if not replaced_key:
            new_lines.append(f"LLM_API_KEY = {llm_api_key}")
        if not replaced_base_url:
            new_lines.append(f"LLM_API_BASE_URL = {llm_api_base_url}")

        self._atomic_write(self.env_path, "\n".join(new_lines).rstrip() + "\n")
        return datetime.now(timezone.utc).isoformat()

    def read_agent_yaml(self) -> dict[str, Any]:
        source = (
            self.agent_yaml_path
            if self.agent_yaml_path.exists()
            else self.agent_yaml_example_path
        )
        if not source.exists():
            return {}

        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def read_agent_yaml_raw(self) -> str:
        source = (
            self.agent_yaml_path
            if self.agent_yaml_path.exists()
            else self.agent_yaml_example_path
        )
        if not source.exists():
            return ""
        return source.read_text(encoding="utf-8")

    def write_agent_yaml(self, data: dict[str, Any]) -> str:
        self.ensure_files_exist()
        self.snapshot_service.create_snapshot(self.agent_yaml_path)

        try:
            content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            raise ValueError(f"invalid yaml payload: {exc}") from exc

        self._atomic_write(self.agent_yaml_path, content)
        return datetime.now(timezone.utc).isoformat()

    def write_agent_yaml_raw(self, raw_yaml: str) -> str:
        self.ensure_files_exist()
        self.snapshot_service.create_snapshot(self.agent_yaml_path)
        try:
            payload = yaml.safe_load(raw_yaml)
        except Exception as exc:
            raise ValueError(f"invalid yaml text: {exc}") from exc

        if payload is not None and not isinstance(payload, dict):
            raise ValueError("yaml root must be a mapping object")

        content = raw_yaml if raw_yaml.endswith("\n") else f"{raw_yaml}\n"
        self._atomic_write(self.agent_yaml_path, content)
        return datetime.now(timezone.utc).isoformat()

    def list_snapshots(self) -> list[str]:
        return self.snapshot_service.list_snapshots()

    def restore_snapshot(self, snapshot_name: str, target: str) -> str:
        if target not in {"env", "agent"}:
            raise ValueError("target must be 'env' or 'agent'")
        target_file = self.env_path if target == "env" else self.agent_yaml_path
        self.snapshot_service.restore_snapshot(snapshot_name, target_file)
        return datetime.now(timezone.utc).isoformat()

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
