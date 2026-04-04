from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from croniter import croniter


def build_timeline(
    timezone: str,
    schedules: list[dict[str, Any]],
    count: int = 10,
) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []

    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    total_count = max(1, min(count, 200))

    for schedule in schedules:
        if not isinstance(schedule, dict) or not schedule.get("enabled", True):
            continue

        name = str(schedule.get("name") or "unnamed")
        kind = str(schedule.get("type") or "cron")

        if kind == "cron":
            expression = str(schedule.get("expression") or "").strip()
            if not expression:
                continue
            try:
                itr = croniter(expression, now)
                for _ in range(total_count):
                    ts = itr.get_next(datetime)
                    events.append(
                        {
                            "schedule_name": name,
                            "trigger_at": ts.isoformat(),
                            "source": "cron",
                        }
                    )
            except Exception:
                continue
        elif kind == "interval":
            try:
                seconds = int(schedule.get("seconds"))
            except (TypeError, ValueError):
                continue
            if seconds <= 0:
                continue
            for i in range(1, total_count + 1):
                ts = now.timestamp() + seconds * i
                events.append(
                    {
                        "schedule_name": name,
                        "trigger_at": datetime.fromtimestamp(ts, tz).isoformat(),
                        "source": "interval",
                    }
                )

    events.sort(key=lambda item: item["trigger_at"])
    return events
