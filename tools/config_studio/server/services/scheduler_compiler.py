from __future__ import annotations

from typing import Any


def _normalize_cron_expression(raw: str) -> str:
    expression = (raw or "").strip()
    if not expression:
        return ""
    parts = expression.split()
    if len(parts) >= 6:
        # UI may provide HH:MM:SS converted to six-field cron; runtime uses five-field.
        return " ".join(parts[1:6])
    return expression


def compile_scheduler_schedules(
    schedules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compile UI schedule structure into runtime DSL-compatible payload.

    Defensive behavior:
    - Preserve original order
    - Ensure required top-level keys exist
    - Keep both legacy steps and new steps_tree when provided
    """
    compiled: list[dict[str, Any]] = []
    for idx, schedule in enumerate(schedules):
        if not isinstance(schedule, dict):
            continue

        name = str(schedule.get("name") or f"schedule_{idx}")
        schedule_type = str(schedule.get("type") or "cron")
        enabled = bool(schedule.get("enabled", True))

        item: dict[str, Any] = {
            "name": name,
            "type": schedule_type,
            "enabled": enabled,
        }

        if schedule_type == "cron":
            item["expression"] = _normalize_cron_expression(
                str(schedule.get("expression") or "")
            )
        elif schedule_type == "interval":
            seconds = schedule.get("seconds")
            try:
                item["seconds"] = int(seconds)
            except (TypeError, ValueError):
                item["seconds"] = 60

        steps = schedule.get("steps")
        if isinstance(steps, list):
            item["steps"] = steps

        steps_tree = schedule.get("steps_tree")
        if isinstance(steps_tree, list):
            item["steps_tree"] = steps_tree

        compiled.append(item)

    return compiled
