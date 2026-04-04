from __future__ import annotations

from .app import app
from .services import build_timeline, compile_scheduler_schedules


def main() -> None:
    """Simple smoke check for Step 1 scaffolding."""
    route_paths = sorted(route.path for route in app.routes)
    print("[SelfTest] Route count:", len(route_paths))
    print("[SelfTest] Contains /api/health:", "/api/health" in route_paths)

    compiled = compile_scheduler_schedules(
        [
            {
                "name": "smoke",
                "type": "cron",
                "expression": "*/5 * * * *",
                "enabled": True,
                "steps_tree": [
                    {
                        "id": "n1",
                        "kind": "action",
                        "action": "core.send_group_msg",
                        "params": {"group_id": "1", "message": "ok"},
                    }
                ],
            }
        ]
    )
    events = build_timeline("Asia/Shanghai", compiled, count=2)
    print("[SelfTest] Compile count:", len(compiled))
    print("[SelfTest] Timeline count:", len(events))


if __name__ == "__main__":
    main()
