from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import paramiko


class DeployService:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def test_connection(
        self,
        host: str,
        port: int,
        username: str,
        auth_type: str,
        password: str | None,
        key_path: str | None,
    ) -> tuple[bool, str, list[str]]:
        logs: list[str] = []
        client = None
        try:
            client = self._connect(host, port, username, auth_type, password, key_path)
            logs.append("ssh connected")
            stdin, stdout, stderr = client.exec_command("echo ok")
            output = stdout.read().decode("utf-8", errors="ignore").strip()
            if output != "ok":
                err = stderr.read().decode("utf-8", errors="ignore").strip()
                logs.append(f"echo mismatch stderr={err}")
                return False, "remote echo check failed", logs
            return True, "connection ok", logs
        except Exception as exc:
            logs.append(f"connection error: {exc}")
            return False, str(exc), logs
        finally:
            if client:
                client.close()

    def push_and_restart(self, payload: dict[str, Any]) -> tuple[bool, str, list[str]]:
        logs: list[str] = []
        host = str(payload.get("host") or "")
        port = int(payload.get("port") or 22)
        username = str(payload.get("username") or "")
        auth_type = str(payload.get("auth_type") or "password")
        password = payload.get("password")
        key_path = payload.get("key_path")
        project_dir = str(payload.get("project_dir") or "/root/qqbot")
        push_env = bool(payload.get("push_env", True))
        push_agent_yaml = bool(payload.get("push_agent_yaml", True))
        restart_policy = str(payload.get("restart_policy") or "docker-compose")

        client = None
        sftp = None
        try:
            client = self._connect(host, port, username, auth_type, password, key_path)
            sftp = client.open_sftp()
            logs.append("sftp opened")

            if push_env:
                local_env = self.repo_root / ".env"
                remote_env = f"{project_dir}/.env"
                self._upload_atomic(sftp, local_env, remote_env)
                logs.append(f"uploaded {local_env} -> {remote_env}")

            if push_agent_yaml:
                local_yaml = self.repo_root / "workflows" / "agent_config.yaml"
                remote_yaml = f"{project_dir}/workflows/agent_config.yaml"
                self._upload_atomic(sftp, local_yaml, remote_yaml)
                logs.append(f"uploaded {local_yaml} -> {remote_yaml}")

            restart_cmd = self._build_restart_command(project_dir, restart_policy)
            if restart_cmd:
                logs.append(f"run restart: {restart_cmd}")
                stdin, stdout, stderr = client.exec_command(restart_cmd)
                code = stdout.channel.recv_exit_status()
                out = stdout.read().decode("utf-8", errors="ignore").strip()
                err = stderr.read().decode("utf-8", errors="ignore").strip()
                if out:
                    logs.append(f"restart stdout: {out}")
                if err:
                    logs.append(f"restart stderr: {err}")
                if code != 0:
                    return False, f"restart failed with code {code}", logs

            return True, "deploy success", logs
        except Exception as exc:
            logs.append(f"deploy error: {exc}")
            return False, str(exc), logs
        finally:
            if sftp:
                sftp.close()
            if client:
                client.close()

    def _connect(
        self,
        host: str,
        port: int,
        username: str,
        auth_type: str,
        password: str | None,
        key_path: str | None,
    ) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if auth_type == "key":
            if not key_path:
                raise ValueError("key_path is required for key auth")
            expanded_key_path = os.path.expanduser(key_path)
            private_key = paramiko.RSAKey.from_private_key_file(expanded_key_path)
            client.connect(
                hostname=host,
                port=port,
                username=username,
                pkey=private_key,
                timeout=10,
            )
        else:
            if not password:
                raise ValueError("password is required for password auth")
            client.connect(
                hostname=host,
                port=port,
                username=username,
                password=password,
                timeout=10,
            )

        return client

    def _upload_atomic(
        self, sftp: paramiko.SFTPClient, local_file: Path, remote_file: str
    ) -> None:
        if not local_file.exists():
            raise FileNotFoundError(f"local file not found: {local_file}")

        remote_tmp = f"{remote_file}.tmp"
        self._ensure_remote_dir(sftp, str(Path(remote_file).parent))
        sftp.put(str(local_file), remote_tmp)
        sftp.rename(remote_tmp, remote_file)

    def _ensure_remote_dir(self, sftp: paramiko.SFTPClient, remote_dir: str) -> None:
        parts = [p for p in remote_dir.split("/") if p]
        current = "/"
        for part in parts:
            current = f"{current}{part}/"
            try:
                sftp.stat(current)
            except IOError:
                sftp.mkdir(current)

    def _build_restart_command(self, project_dir: str, restart_policy: str) -> str:
        if restart_policy == "none":
            return ""
        if restart_policy == "systemctl":
            return "systemctl restart qqbot"
        if restart_policy == "pm2":
            return "pm2 restart qqbot"
        return f"cd {project_dir} && docker compose up -d qqbot || docker compose restart qqbot"
