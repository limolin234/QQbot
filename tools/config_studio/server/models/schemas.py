from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SaveResult(BaseModel):
    success: bool = True
    message: str = "ok"
    updated_at: str


class EnvPatchRequest(BaseModel):
    llm_api_key: str = ""
    llm_api_base_url: str = ""


class AgentPatchRequest(BaseModel):
    data: dict[str, Any]


class AgentRawPatchRequest(BaseModel):
    raw_yaml: str


class AllConfigResponse(BaseModel):
    env: dict[str, str]
    agent: dict[str, Any]


class CompileRequest(BaseModel):
    schedules: list[dict[str, Any]] = Field(default_factory=list)


class TimelineRequest(BaseModel):
    timezone: str = "Asia/Shanghai"
    schedules: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 10


class TimelineEvent(BaseModel):
    schedule_name: str
    trigger_at: str
    source: Literal["cron", "interval"]


class TimelineResponse(BaseModel):
    events: list[TimelineEvent] = Field(default_factory=list)


class DeployTarget(BaseModel):
    host: str
    port: int = 22
    username: str
    auth_type: Literal["password", "key"] = "password"
    password: str | None = None
    key_path: str | None = None
    project_dir: str = "/root/qqbot"


class DeployPushRequest(DeployTarget):
    push_env: bool = True
    push_agent_yaml: bool = True
    restart_policy: Literal["docker-compose", "systemctl", "pm2", "none"] = (
        "docker-compose"
    )


class ConnectionResult(BaseModel):
    success: bool
    message: str
    logs: list[str] = Field(default_factory=list)
