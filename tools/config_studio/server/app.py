from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    AgentPatchRequest,
    AgentRawPatchRequest,
    AllConfigResponse,
    CompileRequest,
    ConnectionResult,
    DeployPushRequest,
    DeployTarget,
    EnvPatchRequest,
    SaveResult,
    TimelineRequest,
    TimelineResponse,
)
from .services import (
    ConfigService,
    DeployService,
    build_timeline,
    compile_scheduler_schedules,
)

APP_TITLE = "QQBot Config Studio API"
APP_VERSION = "0.1.0"


def _detect_repo_root() -> Path:
    current = Path(__file__).resolve()
    # Walk up from tools/config_studio/server/app.py and locate the project root.
    # A valid root must at least contain workflows/ and README.md for this repo.
    for parent in current.parents:
        if (parent / "workflows").exists() and (parent / "README.md").exists():
            return parent
    return current.parents[3]


REPO_ROOT = _detect_repo_root()

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
config_service = ConfigService(REPO_ROOT)
deploy_service = DeployService(REPO_ROOT)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for local Config Studio backend."""
    return {
        "status": "ok",
        "service": "config-studio-server",
        "version": APP_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/config/all", response_model=AllConfigResponse)
async def get_all_config() -> AllConfigResponse:
    try:
        config_service.ensure_files_exist()
        env_map, agent_map = config_service.load_all()
        return AllConfigResponse(env=env_map, agent=agent_map)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to load config: {exc}"
        ) from exc


@app.get("/api/config/agent/raw")
async def get_agent_raw() -> dict[str, str]:
    try:
        config_service.ensure_files_exist()
        return {"raw_yaml": config_service.read_agent_yaml_raw()}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"failed to read raw yaml: {exc}"
        ) from exc


@app.put("/api/config/env", response_model=SaveResult)
async def update_env(payload: EnvPatchRequest) -> SaveResult:
    try:
        updated_at = config_service.write_env(
            payload.llm_api_key, payload.llm_api_base_url
        )
        return SaveResult(updated_at=updated_at)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"failed to update env: {exc}"
        ) from exc


@app.put("/api/config/agent", response_model=SaveResult)
async def update_agent(payload: AgentPatchRequest) -> SaveResult:
    try:
        updated_at = config_service.write_agent_yaml(payload.data)
        return SaveResult(updated_at=updated_at)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"failed to update yaml: {exc}"
        ) from exc


@app.put("/api/config/agent/raw", response_model=SaveResult)
async def update_agent_raw(payload: AgentRawPatchRequest) -> SaveResult:
    try:
        updated_at = config_service.write_agent_yaml_raw(payload.raw_yaml)
        return SaveResult(updated_at=updated_at)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"failed to update raw yaml: {exc}"
        ) from exc


@app.post("/api/scheduler/compile")
async def compile_scheduler(payload: CompileRequest) -> dict[str, Any]:
    try:
        compiled = compile_scheduler_schedules(payload.schedules)
        return {"schedules": compiled}
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"failed to compile scheduler: {exc}"
        ) from exc


@app.post("/api/scheduler/timeline", response_model=TimelineResponse)
async def preview_timeline(payload: TimelineRequest) -> TimelineResponse:
    try:
        events = build_timeline(payload.timezone, payload.schedules, payload.count)
        return TimelineResponse(events=events)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"failed to build timeline: {exc}"
        ) from exc


@app.get("/api/history")
async def list_history() -> dict[str, Any]:
    return {"snapshots": config_service.list_snapshots()}


@app.post("/api/history/restore", response_model=SaveResult)
async def restore_history(payload: dict[str, str]) -> SaveResult:
    snapshot_name = payload.get("snapshot_name", "")
    target = payload.get("target", "")
    if not snapshot_name or not target:
        raise HTTPException(
            status_code=400, detail="snapshot_name and target are required"
        )

    try:
        updated_at = config_service.restore_snapshot(snapshot_name, target)
        return SaveResult(updated_at=updated_at, message="restored")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"failed to restore snapshot: {exc}"
        ) from exc


@app.post("/api/deploy/test-connection", response_model=ConnectionResult)
async def test_connection(payload: DeployTarget) -> ConnectionResult:
    success, message, logs = deploy_service.test_connection(
        host=payload.host,
        port=payload.port,
        username=payload.username,
        auth_type=payload.auth_type,
        password=payload.password,
        key_path=payload.key_path,
    )
    return ConnectionResult(success=success, message=message, logs=logs)


@app.post("/api/deploy/push", response_model=ConnectionResult)
async def deploy_push(payload: DeployPushRequest) -> ConnectionResult:
    success, message, logs = deploy_service.push_and_restart(payload.model_dump())
    return ConnectionResult(success=success, message=message, logs=logs)
