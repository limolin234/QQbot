"""
羽毛球群接龙监控调度器

功能：
- 私聊接收主人的自然语言指令，管理"意向打球时间"（增/删/查）
- 监听指定 QQ 群的消息，用 AI 提取接龙信息
- 将接龙时间与意向时间做碰撞匹配，命中后自动接龙
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, date
from typing import Any

from ncatbot.core import GroupMessage, PrivateMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from bot import bot
from workflows.agent_config_loader import load_current_agent_config

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ─── 工具函数 ──────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now().astimezone()


def _today() -> date:
    return _now().date()


def _log(msg: str) -> None:
    print(f"[BADMINTON_MONITOR] {msg}", flush=True)


def _get_config() -> dict[str, Any]:
    return load_current_agent_config(__file__)


def _get_llm() -> ChatOpenAI:
    cfg = _get_config()
    model = str(cfg.get("model") or "qwen3-max").strip()
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY 环境变量未设置")
    kwargs: dict[str, Any] = {"model": model, "temperature": 0.0, "openai_api_key": api_key}
    base_url = os.getenv("LLM_API_BASE_URL")
    if base_url:
        kwargs["openai_api_base"] = base_url
    return ChatOpenAI(**kwargs)


# ─── 数据库层 ──────────────────────────────────────────────

DB_PATH = os.path.join("data", "badminton_monitor.db")


def _db_connect() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    with _db_connect() as conn:
        # 意向时间段表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT NOT NULL,
                start_date  TEXT,
                end_date    TEXT,
                weekdays    TEXT,
                time_start  TEXT NOT NULL,
                time_end    TEXT NOT NULL,
                note        TEXT DEFAULT '',
                created_at  TEXT NOT NULL
            )
        """)
        # 配置表（关键词、默认名字等）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # 已完成的接龙记录（避免重复接龙）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS completed_signups (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id        TEXT NOT NULL,
                poster_name     TEXT NOT NULL,
                task_date       TEXT NOT NULL,
                task_start      TEXT NOT NULL,
                task_end        TEXT NOT NULL,
                signup_names    TEXT NOT NULL,
                completed_at    TEXT NOT NULL,
                UNIQUE(group_id, poster_name, task_date, task_start, task_end)
            )
        """)
        conn.commit()


def _db_add_schedule(row: dict[str, Any]) -> int:
    with _db_connect() as conn:
        cur = conn.execute(
            """INSERT INTO schedules
               (type, start_date, end_date, weekdays, time_start, time_end, note, created_at)
               VALUES (:type,:start_date,:end_date,:weekdays,:time_start,:time_end,:note,:created_at)""",
            row,
        )
        conn.commit()
        return cur.lastrowid  # type: ignore


def _db_list_schedules() -> list[sqlite3.Row]:
    with _db_connect() as conn:
        return conn.execute("SELECT * FROM schedules ORDER BY id").fetchall()


def _db_delete_schedule(row_id: int) -> bool:
    with _db_connect() as conn:
        cur = conn.execute("DELETE FROM schedules WHERE id=?", (row_id,))
        conn.commit()
        return cur.rowcount > 0


def _db_delete_all_schedules() -> int:
    with _db_connect() as conn:
        cur = conn.execute("DELETE FROM schedules")
        conn.commit()
        return cur.rowcount


def _db_get_config(key: str, default: str = "") -> str:
    with _db_connect() as conn:
        row = conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def _db_set_config(key: str, value: str) -> None:
    with _db_connect() as conn:
        conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value))
        conn.commit()


def _db_is_signup_completed(group_id: str, poster_name: str, task_date: str, task_start: str, task_end: str) -> bool:
    with _db_connect() as conn:
        row = conn.execute(
            """SELECT 1 FROM completed_signups
               WHERE group_id=? AND poster_name=? AND task_date=? AND task_start=? AND task_end=?""",
            (group_id, poster_name, task_date, task_start, task_end),
        ).fetchone()
        return row is not None


def _db_mark_signup_completed(group_id: str, poster_name: str, task_date: str, task_start: str, task_end: str, signup_names: list[str]) -> None:
    with _db_connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO completed_signups
               (group_id, poster_name, task_date, task_start, task_end, signup_names, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (group_id, poster_name, task_date, task_start, task_end, ",".join(signup_names), _now().isoformat()),
        )
        conn.commit()


def _db_list_completed_signups() -> list[sqlite3.Row]:
    with _db_connect() as conn:
        return conn.execute(
            "SELECT * FROM completed_signups ORDER BY task_date DESC, task_start DESC"
        ).fetchall()


def _db_delete_signup(row_id: int) -> bool:
    with _db_connect() as conn:
        cur = conn.execute("DELETE FROM completed_signups WHERE id=?", (row_id,))
        conn.commit()
        return cur.rowcount > 0


def _db_delete_all_signups() -> int:
    with _db_connect() as conn:
        cur = conn.execute("DELETE FROM completed_signups")
        conn.commit()
        return cur.rowcount


def _db_get_expired_signups(now: datetime) -> list[sqlite3.Row]:
    """获取已过期的接龙记录（结束时间+1小时 < 当前时间）"""
    with _db_connect() as conn:
        rows = conn.execute("SELECT * FROM completed_signups").fetchall()

    expired = []
    for row in rows:
        try:
            task_date = date.fromisoformat(row["task_date"])
            end_h, end_m = row["task_end"].split(":")
            end_dt = datetime.combine(task_date, datetime.min.time().replace(hour=int(end_h), minute=int(end_m)))
            end_dt = end_dt.replace(tzinfo=_now().tzinfo)
            # 结束后1小时过期
            expire_dt = end_dt + timedelta(hours=1)
            if now > expire_dt:
                expired.append(row)
        except Exception:
            continue
    return expired


# ─── 时间匹配算法 ──────────────────────────────────────────

def _parse_hm(hm: str) -> int:
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


def _weekday_num(d: date) -> int:
    return d.isoweekday()


def _date_matches_schedule(row: sqlite3.Row, task_date: date) -> bool:
    t = row["type"]
    if t == "once":
        return str(task_date) == row["start_date"]
    if t == "range":
        start = date.fromisoformat(row["start_date"])
        end = date.fromisoformat(row["end_date"])
        return start <= task_date <= end
    if t == "weekly":
        weekdays: list[int] = json.loads(row["weekdays"] or "[]")
        return _weekday_num(task_date) in weekdays
    return False


def _time_overlaps(row: sqlite3.Row, task_start: str, task_end: str) -> bool:
    s1 = _parse_hm(row["time_start"])
    e1 = _parse_hm(row["time_end"])
    s2 = _parse_hm(task_start)
    e2 = _parse_hm(task_end)
    return s1 < e2 and s2 < e1


def find_matching_schedules(task_date: date, task_start: str, task_end: str) -> list[sqlite3.Row]:
    rows = _db_list_schedules()
    matched = []
    for row in rows:
        if _date_matches_schedule(row, task_date) and _time_overlaps(row, task_start, task_end):
            matched.append(row)
    return matched


# ─── AI 调用 ───────────────────────────────────────────────

_PARSE_COMMAND_SYSTEM = """你是一个指令解析助手。用户会用中文描述他们希望监控羽毛球接龙的时间意向。
你需要结合【近期对话历史】和【当前指令】，将意图解析为结构化 JSON。

支持三种类型：
1. once   - 单次，如"11月5日下午2点到4点"
2. range  - 日期范围，如"12月20日到25日全天"
3. weekly - 每周固定，如"每周五到周日18:00-20:00"

输出格式（严格 JSON，不要包裹 markdown 代码块）：
{
  "action": "ADD" | "LIST" | "LIST_SIGNUPS" | "DELETE" | "DELETE_ALL" | "DELETE_SIGNUP" | "DELETE_ALL_SIGNUPS" | "ADD_KEYWORD" | "LIST_KEYWORDS" | "DELETE_KEYWORD" | "SET_DEFAULT_NAME" | "SET_DEFAULT_COUNT" | "UNKNOWN",
  "delete_id": <整数，仅当 action=DELETE 时>,
  "keyword": <字符串，仅当 action=ADD_KEYWORD 或 DELETE_KEYWORD 时>,
  "default_name": <字符串，仅当 action=SET_DEFAULT_NAME 时>,
  "default_count": <整数，仅当 action=SET_DEFAULT_COUNT 时>,
  "schedules": [
    {
      "type": "once" | "range" | "weekly",
      "start_date": "YYYY-MM-DD",
      "end_date":   "YYYY-MM-DD",
      "weekdays":   [1,2,...,7],
      "time_start": "HH:MM",
      "time_end":   "HH:MM",
      "note": "可选备注",
      "signup_count": <整数，本次接龙的人数，默认1>,
      "signup_names": ["名字1", "名字2", ...] <字符串数组，本次接龙的名字列表>
    }
  ]
}

重要规则：
- schedules 是数组，支持一次添加多条
- 若当前指令是对上文的补充，必须结合历史推断完整时间
- 今日日期和近期对话历史会在用户消息中提供
- "全天" → time_start="00:00", time_end="23:59"
- ADD_KEYWORD：添加监控关键词，如"添加"捞"为羽毛球监控关键词"
- DELETE_KEYWORD：删除监控关键词
- SET_DEFAULT_NAME：设置默认接龙名字，如"设置默认接龙名字为张三"
- SET_DEFAULT_COUNT：设置默认接龙人数，如"设置默认接龙人数为2"
- 若无法解析则 action=UNKNOWN
- DELETE 时 delete_id 为整数
- LIST / DELETE_ALL / ADD_KEYWORD / DELETE_KEYWORD / SET_DEFAULT_NAME / SET_DEFAULT_COUNT 时 schedules 可为空数组 []
- LIST_SIGNUPS（查看已接龙）：用户想看自己已经接了哪些场次、打球日程，如"查看已接龙"、"查看打球日程"、"查看当前配置"、"我接了哪些龙"
- LIST（查看配置）：用户想看自己配置了哪些监控时间段，如"查看监控配置"、"我的监控"
- DELETE_SIGNUP：删除某条已接龙记录，如"删除第X条接龙"
- DELETE_ALL_SIGNUPS：清空所有已接龙记录，如"清空所有接龙"、"删除所有日程"
- weekly 的 weekdays：1=周一, 2=周二, 3=周三, 4=周四, 5=周五, 6=周六, 7=周日
- 临时接龙参数：如果用户说"2个人：龚海心、王五"，则 signup_count=2, signup_names=["龚海心", "王五"]
- signup_count 和 signup_names 是临时覆盖，不存入数据库的计划时间配置
"""


_EXTRACT_SIGNUP_SYSTEM = """你是羽毛球接龙信息提取助手。
给定一条群消息（可能是羽毛球捞人接龙帖），你需要提取接龙信息。

输出格式（严格 JSON，不要包裹 markdown 代码块）：
{
  "has_signup": true | false,
  "venue": "场馆名称，如"光体"",
  "task_date": "YYYY-MM-DD",
  "task_start": "HH:MM",
  "task_end": "HH:MM",
  "current_signups": [
    {"position": 1, "name": "张三"},
    {"position": 2, "name": "李四"},
    {"position": 3, "name": ""}  <-- 空表示待填位置
  ],
  "max_positions": 6,
  "summary": "一句话描述接龙内容（50字以内）",
  "poster_name": "发帖人名字（优先使用消息中明确提到的名字，如果没有则使用提供的发帖人昵称）",
  "original_text": "保留原消息中的非接龙部分文字（如场馆描述、费用说明等），一字不动"
}

注意：
- position 是接龙序号，name 为空表示该位置还未有人
- max_positions 是接龙最大人数
- 消息发布时间和今日日期会在用户消息中提供，以供推算相对日期（如"本周六"、"明天"）
- 若消息不包含接龙信息，返回 has_signup=false
- 时间段若只有开始时间，end 可估算（默认+2小时）
- "主日" 指周日；"大后天" 指今日+3天，以此类推
- poster_name：优先使用消息中明确提到的发帖人名字，如果没有则使用提供的发帖人昵称
- original_text 包含原消息中除了接龙列表以外的所有文字，保持原样不动
"""


_GENERATE_SIGNUP_REPLY_SYSTEM = """你是羽毛球接龙助手。给定原消息和接龙信息，你需要生成完整的回复内容。

规则：
1. 保持原消息中除了接龙列表以外的所有文字，一字不动
2. 接龙列表保持原有格式（如"1. 张三\n2. 李四"或"1. 张三 2. 李四"等格式）
3. 把空出的位置填上指定的名字（保持原有格式）
4. 回复开头需要 @ 发帖人
5. 只输出回复内容，不要额外解释

输入格式：
原消息：<original_text>
场馆：<venue>
时间：<task_date> <task_start>-<task_end>
接龙列表：<current_signups 列表，格式为"序号. 名字"，名字为空表示待填位置>
待填写名字：<names_to_fill>
发帖人：<poster_name>

输出：直接输出完整回复内容，不要 JSON，不要 markdown 代码块包裹
"""


async def _call_llm(system: str, user: str) -> str:
    llm = _get_llm()
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    resp = await asyncio.to_thread(llm.invoke, messages)
    return str(resp.content).strip()


async def parse_command(text: str, user_id: str = "") -> dict[str, Any]:
    today_str = _today().isoformat()
    weekday_name = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][_today().weekday()]
    user_msg = f"今日日期：{today_str}（{weekday_name}）\n\n当前指令：{text}"
    raw = await _call_llm(_PARSE_COMMAND_SYSTEM, user_msg)
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception as e:
        _log(f"parse_command json decode error: {e}, raw={raw[:200]}")
        return {"action": "UNKNOWN"}


async def extract_signup_info(msg_text: str, msg_time: datetime, sender_name: str = "") -> dict[str, Any]:
    today_str = _today().isoformat()
    weekday_name = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][_today().weekday()]
    msg_date_str = msg_time.strftime("%Y-%m-%d %H:%M")
    sender_block = f"\n发帖人昵称：{sender_name}" if sender_name else ""
    user_msg = (
        f"今日日期：{today_str}（{weekday_name}）\n"
        f"消息发布时间：{msg_date_str}{sender_block}\n\n"
        f"群消息内容：\n{msg_text}"
    )
    raw = await _call_llm(_EXTRACT_SIGNUP_SYSTEM, user_msg)
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception as e:
        _log(f"extract_signup_info json decode error: {e}, raw={raw[:200]}")
        return {"has_signup": False}


async def generate_signup_reply(
    original_text: str,
    venue: str,
    task_date: str,
    task_start: str,
    task_end: str,
    current_signups: list[dict],
    names_to_fill: list[str],
    poster_name: str,
) -> str:
    signups_str = "\n".join([f"{s['position']}. {s['name'] if s['name'] else '（空）'}" for s in current_signups])
    user_msg = (
        f"原消息：\n{original_text}\n\n"
        f"场馆：{venue}\n"
        f"时间：{task_date} {task_start}-{task_end}\n"
        f"接龙列表：\n{signups_str}\n"
        f"待填写名字：{', '.join(names_to_fill)}\n"
        f"发帖人：{poster_name}"
    )
    raw = await _call_llm(_GENERATE_SIGNUP_REPLY_SYSTEM, user_msg)
    raw = raw.strip().lstrip("```").rstrip("```").strip()
    return raw


# ─── 格式化工具 ────────────────────────────────────────────

_WEEKDAY_NAMES = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}


def _format_schedule_row(row: sqlite3.Row) -> str:
    t = row["type"]
    ts = row["time_start"]
    te = row["time_end"]
    note = row["note"] or ""
    type_label = {"once": "单次", "range": "区间", "weekly": "每周"}.get(t, t)
    if t == "once":
        date_part = row["start_date"]
    elif t == "range":
        date_part = f"{row['start_date']} ~ {row['end_date']}"
    else:
        days = json.loads(row["weekdays"] or "[]")
        day_names = "、".join(_WEEKDAY_NAMES.get(d, str(d)) for d in sorted(days))
        date_part = day_names
    note_part = f"  📝{note}" if note else ""
    return f"#{row['id']} [{type_label}] {date_part}  {ts}–{te}{note_part}"


def format_schedule_list(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return "当前没有任何打球时间配置。"

    # 获取已接龙记录
    signups = _db_list_completed_signups()

    # 按 schedule_id 分组（暂时只按时间段匹配）
    lines = ["🏸 当前监控的打球时间配置："]
    for row in rows:
        row_id = row["id"]
        t = row["type"]
        ts = row["time_start"]
        te = row["time_end"]

        # 查找该时间段的已接龙记录
        matched_signups = []
        for s in signups:
            if s["task_start"] == ts and s["task_end"] == te:
                matched_signups.append(s)

        # 状态 emoji
        if matched_signups:
            status = "✅"  # 已有接龙
            signup_info = " | ".join([
                f"{s['task_date']} 接龙:{s['signup_names']}"
                for s in matched_signups
            ])
        else:
            status = "⏳"  # 待接龙
            signup_info = ""

        note = row["note"] or ""
        note_part = f"  📝{note}" if note else ""

        type_label = {"once": "单次", "range": "区间", "weekly": "每周"}.get(t, t)
        if t == "once":
            date_part = row["start_date"]
        elif t == "range":
            date_part = f"{row['start_date']} ~ {row['end_date']}"
        else:
            days = json.loads(row["weekdays"] or "[]")
            day_names = "、".join(_WEEKDAY_NAMES.get(d, str(d)) for d in sorted(days))
            date_part = day_names

        line = f"#{row_id} {status} [{type_label}] {date_part}  {ts}–{te}{note_part}"
        if signup_info:
            line += f"\n   └ {signup_info}"
        lines.append(line)

    return "\n".join(lines)


# ─── 主调度器类 ────────────────────────────────────────────

class BadmintonMonitorScheduler:
    def __init__(self) -> None:
        _ensure_table()
        _log("Scheduler initialized.")

    def start_up(self) -> None:
        cfg = _get_config()
        _log(f"Configured monitor groups: {cfg.get('monitor_groups', [])}")
        # 启动时清理过期接龙
        asyncio.create_task(self._cleanup_expired_signups())

    async def _cleanup_expired_signups(self) -> None:
        """定期清理过期的接龙记录（结束后1小时自动删除）"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时检查一次
                expired = _db_get_expired_signups(_now())
                if expired:
                    admin_qq = str(_get_config().get("admin_qq") or "").strip()
                    deleted_info = []
                    for row in expired:
                        deleted_info.append(
                            f"• {row['task_date']} {row['task_start']}-{row['task_end']} "
                            f"接龙:{row['signup_names']}"
                        )
                        _db_delete_signup(row["id"])
                    if admin_qq and deleted_info:
                        await self._send_private(
                            admin_qq,
                            f"🗑️ 以下接龙日程已过期（结束后超过1小时），已自动清理：\n" +
                            "\n".join(deleted_info) +
                            "\n\n— [羽毛球监控]"
                        )
                    _log(f"Cleaned up {len(expired)} expired signups")
            except Exception as e:
                _log(f"_cleanup_expired_signups error: {e}")

    async def handle_private_command(self, msg: PrivateMessage) -> bool:
        cfg = _get_config()
        admin_qq = str(cfg.get("admin_qq") or "").strip()
        user_id = str(getattr(msg, "user_id", "") or "").strip()

        if not admin_qq or user_id != admin_qq:
            return False

        text = str(getattr(msg, "raw_message", "") or "").strip()
        if not text:
            return False

        lower = text.lower()
        is_cmd = any(k in lower for k in [
            "监控", "打球", "羽毛球", "添加", "删除", "查看", "清空",
            "配置", "关键词", "接龙", "时间段", "设置默认",
        ])
        if not is_cmd:
            return False

        _log(f"Private command from admin {user_id}: {text[:80]}")
        parsed = await parse_command(text, user_id=user_id)
        action = str(parsed.get("action") or "UNKNOWN").upper()
        _log(f"Parsed action={action} parsed={json.dumps(parsed, ensure_ascii=False)[:200]}")

        if action == "LIST":
            rows = _db_list_schedules()
            reply = format_schedule_list(rows)

        elif action == "LIST_SIGNUPS":
            signups = _db_list_completed_signups()
            if not signups:
                reply = "目前没有任何已接龙的记录。"
            else:
                lines = ["🏸 已接龙的打球日程："]
                for row in signups:
                    names = row["signup_names"] or ""
                    weekday = _WEEKDAY_NAMES.get(_weekday_num(date.fromisoformat(row["task_date"])), "")
                    lines.append(
                        f"• {row['task_date']}（{weekday}）{row['task_start']}-{row['task_end']} "
                        f"| 接龙：{names} | 发帖：{row['poster_name']}"
                    )
                reply = "\n".join(lines)

        elif action == "DELETE_SIGNUP":
            del_id = parsed.get("delete_id")
            if del_id is None:
                reply = '⚠️ 未识别到要删除的接龙编号，请说"删除接龙第X条"。'
            else:
                ok = _db_delete_signup(int(del_id))
                reply = f"✅ 已删除接龙第 {del_id} 条。" if ok else f"⚠️ 未找到编号 {del_id} 的接龙记录。"

        elif action == "DELETE_ALL_SIGNUPS":
            count = _db_delete_all_signups()
            reply = f"✅ 已清空所有接龙记录（共删除 {count} 条）。"

        elif action == "DELETE_ALL":
            count = _db_delete_all_schedules()
            reply = f"✅ 已清空所有打球时间配置（共删除 {count} 条）。"

        elif action == "DELETE":
            del_id = parsed.get("delete_id")
            if del_id is None:
                reply = '⚠️ 未识别到要删除的编号，请说"删除第N条"。'
            else:
                ok = _db_delete_schedule(int(del_id))
                reply = f"✅ 已删除第 {del_id} 条配置。" if ok else f"⚠️ 未找到编号 {del_id} 的配置。"

        elif action == "ADD":
            scheds = parsed.get("schedules") or []
            if not scheds:
                reply = "⚠️ 解析失败：未识别到有效的时间配置，请重新描述。"
            else:
                added_descs = []
                for sched in scheds:
                    t = str(sched.get("type") or "").strip()
                    if t not in ("once", "range", "weekly"):
                        continue
                    weekdays_raw = sched.get("weekdays") or []
                    signup_count = sched.get("signup_count")
                    signup_names = sched.get("signup_names") or []

                    # 临接参数编码到 note 中
                    note_parts = []
                    user_note = str(sched.get("note") or "").strip()
                    if user_note:
                        note_parts.append(user_note)
                    if signup_count is not None:
                        names_str = "、".join(signup_names) if signup_names else ""
                        note_parts.append(f"临接:{signup_count}人{'|'+names_str if names_str else ''}")

                    note = " | ".join(note_parts) if note_parts else ""
                    row = {
                        "type": t,
                        "start_date": sched.get("start_date") or None,
                        "end_date": sched.get("end_date") or None,
                        "weekdays": json.dumps(weekdays_raw) if weekdays_raw else "[]",
                        "time_start": str(sched.get("time_start") or "00:00"),
                        "time_end": str(sched.get("time_end") or "23:59"),
                        "note": note,
                        "created_at": _now().isoformat(),
                    }
                    new_id = _db_add_schedule(row)
                    with _db_connect() as conn:
                        new_row = conn.execute("SELECT * FROM schedules WHERE id=?", (new_id,)).fetchone()
                    added_descs.append(_format_schedule_row(new_row) if new_row else str(new_id))
                if added_descs:
                    reply = "✅ 已添加打球时间配置：\n" + "\n".join(added_descs)
                else:
                    reply = "⚠️ 解析失败：时间类型无效，请重新描述。"

        elif action == "ADD_KEYWORD":
            kw = str(parsed.get("keyword") or "").strip()
            if not kw:
                reply = "⚠️ 未识别到关键词，请说\"添加xxx为羽毛球监控关键词\"。"
            else:
                existing = _db_get_config("keywords", "[]")
                keywords = json.loads(existing) if existing else []
                if kw in keywords:
                    reply = f"⚠️ 关键词「{kw}」已存在。"
                else:
                    keywords.append(kw)
                    _db_set_config("keywords", json.dumps(keywords, ensure_ascii=False))
                    reply = f"✅ 已添加关键词「{kw}」，当前关键词：{'、'.join(keywords)}"

        elif action == "DELETE_KEYWORD":
            kw = str(parsed.get("keyword") or "").strip()
            if not kw:
                reply = "⚠️ 未识别到关键词，请说\"删除xxx羽毛球监控关键词\"。"
            else:
                existing = _db_get_config("keywords", "[]")
                keywords = json.loads(existing) if existing else []
                if kw in keywords:
                    keywords.remove(kw)
                    _db_set_config("keywords", json.dumps(keywords, ensure_ascii=False))
                    reply = f"✅ 已删除关键词「{kw}」，当前关键词：{'、'.join(keywords) if keywords else '（无）'}"
                else:
                    reply = f"⚠️ 关键词「{kw}」不存在。"

        elif action == "LIST_KEYWORDS":
            existing = _db_get_config("keywords", "[]")
            keywords = json.loads(existing) if existing else []
            default_name = _db_get_config("default_name", "龚海心")
            default_count = _db_get_config("default_count", "1")
            reply = (
                f"🏸 羽毛球监控配置：\n"
                f"• 关键词：{'、'.join(keywords) if keywords else '（无）'}\n"
                f"• 默认接龙名字：{default_name}\n"
                f"• 默认接龙人数：{default_count}"
            )

        elif action == "SET_DEFAULT_NAME":
            name = str(parsed.get("default_name") or "").strip()
            if not name:
                reply = "⚠️ 未识别到名字，请说\"设置默认接龙名字为张三\"。"
            else:
                _db_set_config("default_name", name)
                reply = f"✅ 已设置默认接龙名字为「{name}」"

        elif action == "SET_DEFAULT_COUNT":
            count = parsed.get("default_count")
            if count is None:
                reply = "⚠️ 未识别到人数，请说\"设置默认接龙人数为2\"。"
            else:
                _db_set_config("default_count", str(int(count)))
                reply = f"✅ 已设置默认接龙人数为 {count}"

        else:
            reply = (
                "❓ 未识别的指令。\n"
                "• \"帮我监控每周五18-20点打球\"\n"
                "• \"查看监控\"\n"
                "• \"删除第1条\"\n"
                "• \"添加\"捞\"为羽毛球监控关键词\"\n"
                "• \"设置默认接龙名字为张三\"\n"
                "• \"设置默认接龙人数为2\""
            )

        _log(f"Sending reply to {admin_qq}, length={len(reply)}")
        await self._send_private(admin_qq, reply + "\n\n— [羽毛球监控]")
        return True

    async def handle_group_message(self, msg: GroupMessage) -> None:
        cfg = _get_config()
        monitor_groups = cfg.get("monitor_groups") or []
        if isinstance(monitor_groups, str):
            monitor_groups = [monitor_groups]
        monitor_groups = [str(g).strip() for g in monitor_groups]

        group_id = str(getattr(msg, "group_id", "") or "").strip()
        if group_id not in monitor_groups:
            return

        text = str(getattr(msg, "raw_message", "") or "").strip()
        if not text or len(text) < 5:
            return

        # 关键词筛选
        existing = _db_get_config("keywords", '["捞","接龙"]')
        keywords = json.loads(existing) if existing else ["捞", "接龙"]
        if not any(kw in text for kw in keywords):
            return

        _log(f"Group {group_id} keyword hit, extracting signup info...")

        # 获取发帖人昵称
        sender = getattr(msg, "sender", None)
        sender_name = str(getattr(sender, "nickname", "") or "").strip()
        if not sender_name:
            sender_name = str(getattr(msg, "user_name", "") or "").strip()
        if not sender_name:
            sender_name = str(getattr(msg, "user_id", "") or "").strip()

        msg_time = _now()
        result = await extract_signup_info(text, msg_time, sender_name=sender_name)

        if not result.get("has_signup"):
            return

        task_date_str = str(result.get("task_date") or "").strip()
        task_start = str(result.get("task_start") or "").strip()
        task_end = str(result.get("task_end") or "").strip()
        # 优先使用消息中提取的poster_name，否则用发送者的昵称
        extracted_poster = str(result.get("poster_name") or "").strip()
        poster_name = extracted_poster if extracted_poster else sender_name

        if not task_date_str or not task_start or not task_end:
            _log(f"Incomplete task time extraction: {result}")
            return

        try:
            task_date = date.fromisoformat(task_date_str)
        except ValueError:
            _log(f"Invalid task_date: {task_date_str}")
            return

        _log(f"Signup extracted: date={task_date} {task_start}-{task_end}, poster={poster_name}")

        # 检查是否已经接龙过了
        if _db_is_signup_completed(group_id, poster_name, task_date_str, task_start, task_end):
            _log(f"Signup already completed for {poster_name} at {task_date_str} {task_start}-{task_end}, skipping")
            admin_qq = str(cfg.get("admin_qq") or "").strip()
            if admin_qq:
                await self._send_private(
                    admin_qq,
                    f"⚠️ 已跳过重复接龙：{poster_name} 在 {task_date_str} {task_start}-{task_end} 的接龙你之前已接过了\n\n— [羽毛球监控]",
                )
            return

        matched = find_matching_schedules(task_date, task_start, task_end)
        if not matched:
            _log(f"No matching schedules found for {task_date} {task_start}-{task_end}")
            return

        # 从第一个匹配的 schedule 中获取配置（优先使用 note 中的临接参数）
        schedule = matched[0]
        schedule_note = str(schedule["note"] or "")

        # 解析临接参数（格式："临接:2人|龚海心、王五"）
        signup_count_override = None
        signup_names_override = None
        if "临接:" in schedule_note:
            parts = schedule_note.split("临接:")[1].split("|")
            count_part = parts[0].replace("人", "").strip()
            try:
                signup_count_override = int(count_part)
            except ValueError:
                pass
            if len(parts) > 1 and parts[1].strip():
                signup_names_override = [n.strip() for n in parts[1].split("、") if n.strip()]

        # 获取配置（临接参数优先，否则用全局默认）
        default_name = _db_get_config("default_name", "龚海心")
        default_count = int(_db_get_config("default_count", "1"))

        use_count = signup_count_override if signup_count_override is not None else default_count
        use_names = signup_names_override if signup_names_override else [default_name]

        # 找到空位
        current_signups = result.get("current_signups", [])
        empty_positions = [s for s in current_signups if not s.get("name")]

        if not empty_positions:
            _log("No empty positions available")
            return

        # 检查是否有足够的空位
        if len(empty_positions) < use_count:
            _log(f"Not enough empty positions ({len(empty_positions)}) for use_count ({use_count})")
            return

        # 构建接龙名字列表
        names_to_fill = []
        for i in range(use_count):
            name = use_names[i] if i < len(use_names) else use_names[-1]
            names_to_fill.append(name)

        # 标记为已完成（在发送前，避免并发问题）
        _db_mark_signup_completed(group_id, poster_name, task_date_str, task_start, task_end, names_to_fill)

        # 用 AI 生成完整回复（保持原消息格式不变）
        venue = str(result.get("venue") or "")
        reply_text = await generate_signup_reply(
            original_text=text,  # 传入原始消息，让 AI 来判别保留哪些部分
            venue=venue,
            task_date=task_date_str,
            task_start=task_start,
            task_end=task_end,
            current_signups=current_signups,
            names_to_fill=names_to_fill,
            poster_name=poster_name,
        )

        _log(f"Posting signup reply to group {group_id}")
        await self._send_group(group_id, reply_text)

        # 通知管理员
        admin_qq = str(cfg.get("admin_qq") or "").strip()
        if admin_qq:
            await self._send_private(
                admin_qq,
                f"✅ 已接龙：{poster_name} 在 {task_date_str} {task_start}-{task_end} 的接龙\n"
                f"接龙位置：{', '.join(names_to_fill)}\n\n— [羽毛球监控]",
            )

    async def _send_private(self, qq: str, text: str) -> None:
        try:
            await bot.api.post_private_msg(qq, text=text)
        except Exception as e:
            _log(f"send_private failed to {qq}: {e}")

    async def _send_group(self, group_id: str, text: str) -> None:
        try:
            await bot.api.post_group_msg(group_id, text=text)
        except Exception as e:
            _log(f"send_group failed to {group_id}: {e}")


badminton_monitor_scheduler = BadmintonMonitorScheduler()
