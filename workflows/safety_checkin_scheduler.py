"""
Safety Check-in 报平安调度器。

职责：
1. 每天 21:00 自动在监控群发起报平安接龙
2. 每隔 interval_minutes 分钟检查并升级提醒（测试3轮，正式6轮，22:00 截止）
3. 严格匹配关键词 "1" 识别接龙
4. 全员完成或超时时汇报给主人（含 QQ 号）

提醒升级策略：
  第1-2轮：@all 全体提醒
  第3轮起：@ 未接龙的人
  结束时  ：汇报主人（完成/未完成名单+QQ号）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, TypedDict

import aiocron

from bot import QQnumber, bot
from ncatbot.core.event.message_segment.message_array import MessageArray
from workflows.agent_config_loader import load_current_agent_config
from workflows.agent_observe import bind_agent_event

# ─────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────

SAFETY_CONFIG = load_current_agent_config(__file__)

# ─────────────────────────────────────────────
# 数据库
# ─────────────────────────────────────────────

DB_PATH = SAFETY_CONFIG.get("db_path", "data/safety_checkin.db")


def _db_connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    with _db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkin_records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                group_id    TEXT NOT NULL,
                member_qq   TEXT NOT NULL,
                member_name TEXT NOT NULL,
                checked_in  INTEGER DEFAULT 0,
                round       INTEGER DEFAULT 0,
                checked_at  TEXT,
                UNIQUE(date, group_id, member_qq)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkin_session (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL UNIQUE,
                group_id    TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                ended_at    TEXT,
                final_round INTEGER,
                status      TEXT DEFAULT 'active',
                report      TEXT
            )
        """)
        conn.commit()


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _today() -> str:
    return _now().strftime("%Y-%m-%d")


def _members() -> dict[str, str]:
    """返回 {qq: name} 映射。"""
    raw = SAFETY_CONFIG.get("members") or {}
    if isinstance(raw, dict):
        return {str(v).strip(): str(k).strip() for k, v in raw.items() if k and v}
    return {}


def _monitor_groups() -> list[str]:
    groups = SAFETY_CONFIG.get("monitor_groups") or []
    return [str(g).strip() for g in groups if str(g).strip()]


def _log(msg: str) -> None:
    print(f"[SafetyCheckin] {msg}")


def _read_messages_since(path: str, since_ts: str) -> list[dict]:
    """从 message.jsonl 读取 >= since_ts 的群消息。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

    keyword = SAFETY_CONFIG.get("checkin_keyword", "1")
    results = []
    for line in lines:
        if line.get("chat_type") != "group":
            continue
        msg_ts = str(line.get("ts", ""))
        if msg_ts < since_ts:
            continue
        msg_text = str(line.get("cleaned_message", "")).strip()
        if msg_text == keyword:
            results.append(line)
    return results


def _send_group_msg(group_id: str, text: str) -> None:
    try:
        # 同步调用，在 async 环境外使用
        loop = asyncio.get_event_loop()
        loop.run_until_complete(bot.api.post_group_msg(group_id, text=text))
    except Exception as e:
        _log(f"send_group_msg failed: {e}")


async def _send_group_msg_async(group_id: str, text: str) -> None:
    try:
        _log(f"[DEBUG] _send_group_msg_async called: group={group_id}, text={text[:50]}")
        result = await asyncio.wait_for(bot.api.post_group_msg(group_id, text=text), timeout=10.0)
        _log(f"[DEBUG] _send_group_msg_async success: {result}")
    except asyncio.TimeoutError:
        _log(f"[ERROR] send_group_msg_async timeout: group {group_id}")
    except Exception as e:
        _log(f"[ERROR] send_group_msg_async failed: {e}")


async def _send_private_msg(user_id: str, text: str) -> None:
    try:
        await asyncio.wait_for(bot.api.post_private_msg(user_id, text=text), timeout=10.0)
    except asyncio.TimeoutError:
        _log(f"send_private_msg timeout: user {user_id}")
    except Exception as e:
        _log(f"send_private_msg failed: {e}")
    except Exception as e:
        _log(f"send_private_msg failed: {e}")


# ─────────────────────────────────────────────
# 数据库读写
# ─────────────────────────────────────────────

def _init_session(group_id: str) -> None:
    today = _today()
    _ensure_table()
    with _db_connect() as conn:
        # 写入/更新成员
        members = _members()
        for qq, name in members.items():
            conn.execute("""
                INSERT INTO checkin_records (date, group_id, member_qq, member_name, checked_in, round)
                VALUES (?, ?, ?, ?, 0, 0)
                ON CONFLICT(date, group_id, member_qq)
                    DO UPDATE SET checked_in=0, round=0, checked_at=NULL
            """, (today, group_id, qq, name))

        # 创建 session
        now_iso = _now().isoformat(timespec="seconds")
        conn.execute("""
            INSERT OR REPLACE INTO checkin_session
                (date, group_id, started_at, ended_at, final_round, status, report)
            VALUES (?, ?, ?, NULL, 0, 'active', NULL)
        """, (today, group_id, now_iso))
        conn.commit()


def _mark_checked_in(member_qq: str, group_id: str, current_round: int) -> bool:
    """标记成员已接龙，返回是否是新标记（避免重复）。"""
    today = _today()
    with _db_connect() as conn:
        cur = conn.execute("""
            SELECT checked_in FROM checkin_records
            WHERE date=? AND group_id=? AND member_qq=?
        """, (today, group_id, member_qq))
        row = cur.fetchone()
        if row and row["checked_in"] == 1:
            return False  # 已接龙，跳过
        conn.execute("""
            UPDATE checkin_records
            SET checked_in=1, round=?, checked_at=?
            WHERE date=? AND group_id=? AND member_qq=?
        """, (current_round, _now().isoformat(timespec="seconds"), today, group_id, member_qq))
        conn.commit()
        return True


def _get_pending_members(group_id: str) -> list[tuple[str, str]]:
    """返回未接龙成员列表 [(qq, name), ...]。"""
    today = _today()
    with _db_connect() as conn:
        rows = conn.execute("""
            SELECT member_qq, member_name FROM checkin_records
            WHERE date=? AND group_id=? AND checked_in=0
        """, (today, group_id)).fetchall()
        return [(str(r["member_qq"]), str(r["member_name"])) for r in rows]


def _get_checked_count(group_id: str) -> tuple[int, int]:
    """返回 (已接龙数, 总数)。"""
    today = _today()
    with _db_connect() as conn:
        total = conn.execute("""
            SELECT COUNT(*) FROM checkin_records WHERE date=? AND group_id=?
        """, (today, group_id)).fetchone()[0]
        done = conn.execute("""
            SELECT COUNT(*) FROM checkin_records WHERE date=? AND group_id=? AND checked_in=1
        """, (today, group_id)).fetchone()[0]
        return done, total


def _end_session(group_id: str, status: str, report: str, final_round: int) -> None:
    """结束会话。"""
    today = _today()
    with _db_connect() as conn:
        conn.execute("""
            UPDATE checkin_session
            SET ended_at=?, status=?, report=?, final_round=?
            WHERE date=? AND group_id=? AND status='active'
        """, (_now().isoformat(timespec="seconds"), status, report, final_round, today, group_id))
        conn.commit()


def _get_session_start_ts(group_id: str) -> str:
    """获取当前 session 的启动时间戳（ISO）。"""
    today = _today()
    with _db_connect() as conn:
        row = conn.execute("""
            SELECT started_at FROM checkin_session WHERE date=? AND group_id=?
        """, (today, group_id)).fetchone()
        return str(row["started_at"]) if row else ""


def _get_last_check_time() -> str:
    """返回上次检查时间的 ISO 字符串（用于过滤 message.jsonl）。"""
    return _now().isoformat(timespec="seconds")


# ─────────────────────────────────────────────
# LangGraph 状态定义
# ─────────────────────────────────────────────

class SafetyCheckinState(TypedDict, total=False):
    group_id: str
    target_qqs: dict[str, str]         # qq -> name
    round: int                            # 当前轮次 1-6
    checked_in: list[str]                # 已接龙 QQ 列表
    pending: list[str]                   # 未接龙 QQ 列表
    reminder_sent_this_round: bool         # 本轮是否已发提醒
    done: bool                            # 任务是否结束
    report_text: str                      # 最终汇报文本
    start_time: str                       # ISO 时间戳
    deadline_time: str                     # ISO 时间戳截止时间
    interval_seconds: int                 # 提醒间隔秒数
    checkin_keyword: str                  # 接龙关键词
    last_check_time: str                  # 上次检查时间（ISO）
    escalation_type: str                   # 当前升级方式: at_all / at_person


# ─────────────────────────────────────────────
# LangGraph 节点
# ─────────────────────────────────────────────

def _build_reminder_text(pending: list[str], target_qqs: dict[str, str],
                         escalation_type: str, round_num: int) -> str:
    """根据升级类型构建提醒文本，@和文字分开展示（分两行）。"""
    if escalation_type == "at_all":
        # @all + 纯文字提醒（分两行）
        return f"[CQ:at,qq=all]\n请各位同学回复「1」完成报平安接龙"
    elif escalation_type == "at_person":
        # @个人 + 纯文字提醒（分两行）
        names = [target_qqs.get(qq, qq) for qq in pending]
        at_parts = [f"[CQ:at,qq={qq}]" for qq in pending]
        at_text = "".join(at_parts)
        return f"{at_text}\n请以上同学回复「1」完成报平安接龙：{', '.join(names)}"
    else:
        return ""


async def _send_reminder(state: SafetyCheckinState) -> SafetyCheckinState:
    """发送本轮提醒。"""
    if state["reminder_sent_this_round"]:
        return state  # 本轮已发过，跳过

    pending = state["pending"]
    if not pending:
        return state

    escalation = state["escalation_type"]
    group_id = state["group_id"]
    round_num = state["round"]
    target_qqs = state["target_qqs"]

    text = _build_reminder_text(pending, target_qqs, escalation, round_num)
    if text:
        await _send_group_msg_async(group_id, text)
        _log(f"Round {round_num} reminder sent ({escalation}) to group {group_id}")

    return {**state, "reminder_sent_this_round": True}


async def _check_messages(state: SafetyCheckinState) -> SafetyCheckinState:
    """
    从数据库读取当前报到状态（替代读 message.jsonl）。
    实时反映 group_entrance 写入的最新数据。
    """
    group_id = state["group_id"]
    target_qqs = state["target_qqs"]

    # 直接从数据库读取最新 pending 列表
    pending_db = _get_pending_members(group_id)
    pending_qqs = [qq for qq, _ in pending_db]
    checked_qqs = [qq for qq in target_qqs if qq not in pending_qqs]

    if checked_qqs:
        names = [target_qqs.get(q, q) for q in checked_qqs]
        _log(f"Checked in DB: {names}")

    return {
        **state,
        "checked_in": checked_qqs,
        "pending": pending_qqs,
        "last_check_time": _now().isoformat(timespec="seconds"),
    }


def _check_deadline(state: SafetyCheckinState) -> SafetyCheckinState:
    """判断是否超时（到达截止时间）。"""
    deadline_str = state["deadline_time"]
    now = _now()
    try:
        from datetime import datetime
        deadline = datetime.fromisoformat(deadline_str)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        deadline = now

    timed_out = now >= deadline
    return {**state}


def _escalation_decision(state: SafetyCheckinState) -> SafetyCheckinState:
    """根据轮次决定升级方式。"""
    round_num = state["round"]
    pending = state["pending"]

    if not pending:
        escalation = "done"
    elif round_num <= 2:
        escalation = "at_all"
    elif round_num <= 4:
        escalation = "at_person"
    else:
        escalation = "at_person"  # 第5-6轮改为群内@个人（私聊有非好友限制）

    return {**state, "escalation_type": escalation, "reminder_sent_this_round": False}


def _check_completion(state: SafetyCheckinState) -> SafetyCheckinState:
    """检查是否全员完成。"""
    pending = state["pending"]
    done = not pending
    return {**state, "done": done}


def _report_completed(state: SafetyCheckinState) -> SafetyCheckinState:
    """全员完成，生成汇报。"""
    group_id = state["group_id"]
    target_qqs = state["target_qqs"]
    checked = state["checked_in"]
    now = _now().strftime("%Y-%m-%d %H:%M")

    checked_detail = [f"{target_qqs.get(q, q)}({q})" for q in checked]
    report = (
        f"【报平安完成】\n"
        f"时间：{now}\n"
        f"群号：{group_id}\n"
        f"应报到：{len(target_qqs)} 人\n"
        f"实报到：{len(checked)} 人\n"
        f"已报到：{', '.join(checked_detail)}"
    )
    return {**state, "report_text": report, "done": True}


def _report_timeout(state: SafetyCheckinState) -> SafetyCheckinState:
    """超时，生成未完成名单汇报（含 QQ 号）。"""
    group_id = state["group_id"]
    target_qqs = state["target_qqs"]
    checked = state["checked_in"]
    pending = state["pending"]
    now = _now().strftime("%Y-%m-%d %H:%M")

    checked_names = [target_qqs.get(q, q) for q in checked]
    pending_detail = [f"{target_qqs.get(q, q)}({q})" for q in pending]

    report = (
        f"【报平安超时】\n"
        f"时间：{now}\n"
        f"群号：{group_id}\n"
        f"应报到：{len(target_qqs)} 人\n"
        f"实报到：{len(checked)} 人\n"
        f"未报到（{len(pending)} 人）：{', '.join(pending_detail)}"
    )
    return {**state, "report_text": report, "done": True}


# ─────────────────────────────────────────────
# 简化版状态更新（替代 LangGraph 循环）
# ─────────────────────────────────────────────

def _update_state_for_next_round(state: SafetyCheckinState) -> SafetyCheckinState:
    """在每轮结束时调用，计算下一轮状态。escalation_type 由调用方传入。"""
    pending = state["pending"]
    done = not pending
    return {
        **state,
        "reminder_sent_this_round": False,
        "done": done,
    }


# ─────────────────────────────────────────────
# Scheduler 主循环
# ─────────────────────────────────────────────

class SafetyCheckinScheduler:
    def __init__(self) -> None:
        _ensure_table()
        _log("Scheduler initialized.")

    def start_up(self) -> None:
        cfg = SAFETY_CONFIG
        _log(f"Configured monitor groups: {cfg.get('monitor_groups', [])}")
        _log(f"Members: {cfg.get('members', {})}")
        # 每天 21:00 触发
        aiocron.crontab("0 21 * * *", func=self._start_daily_cycle)

    def _start_daily_cycle(self) -> None:
        _log("Triggering daily check-in cycle...")
        asyncio.create_task(self._run_cycle())

    async def _run_cycle(self, *, test_mode: bool = False) -> None:
        """执行一轮完整的报平安流程。"""
        groups = _monitor_groups()
        _log(f"[DEBUG] _run_cycle started, test_mode={test_mode}, groups={groups}")
        if not groups:
            _log("No monitor groups configured.")
            return

        cfg = SAFETY_CONFIG
        start_hour = int(cfg.get("start_hour", 21))
        end_hour = int(cfg.get("end_hour", 22))
        interval_minutes = int(cfg.get("interval_minutes", 10))
        interval_seconds = interval_minutes * 60
        keyword = str(cfg.get("checkin_keyword", "1"))

        now = _now()
        start_time = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        deadline_time = now.replace(hour=end_hour, minute=0, second=0, microsecond=0)

        if test_mode:
            # 测试模式：缩短为 3 秒间隔，6 轮总共 18 秒
            interval_seconds = 3
            deadline_time = now.replace(hour=23, minute=59, second=59)
            _log("TEST MODE: interval=3s, deadline extended")
        else:
            interval_seconds = interval_minutes * 60

        target_qqs = _members()  # {qq: name}
        if not target_qqs:
            _log("No members configured.")
            return

        group_id = groups[0]  # 目前只支持第一个群

        # 初始化数据库 session
        _init_session(group_id)
        _log(f"Session started for group {group_id}")

        # 发送第一轮 @all 提醒（@all 和文字分两条消息发送）
        now_iso = _now().isoformat(timespec="seconds")
        _log(f"[DEBUG] About to send initial @all to group {group_id}")
        try:
            at_all_msg = MessageArray()
            at_all_msg.add_at_all()
            await asyncio.wait_for(bot.api.post_group_array_msg(group_id, at_all_msg), timeout=10.0)
            await asyncio.wait_for(bot.api.post_group_msg(group_id, text="报平安接龙开始！请回复「1」完成接龙"), timeout=10.0)
            _log(f"[DEBUG] Initial @all sent to group {group_id}")
        except Exception as e:
            _log(f"[WARN] Initial @all failed ({e}), sending fallback")
            try:
                await asyncio.wait_for(bot.api.post_group_msg(group_id, text="【报平安提醒】请回复「1」完成接龙"), timeout=10.0)
            except Exception:
                pass

        # 状态初始化
        initial_state: SafetyCheckinState = {
            "group_id": group_id,
            "target_qqs": target_qqs,
            "round": 1,
            "checked_in": [],
            "pending": list(target_qqs.keys()),
            "reminder_sent_this_round": False,
            "done": False,
            "report_text": "",
            "start_time": now_iso,
            "deadline_time": deadline_time.isoformat(timespec="seconds"),
            "interval_seconds": interval_seconds,
            "checkin_keyword": keyword,
            "last_check_time": now_iso,
            "escalation_type": "at_all",
        }

        max_rounds = int(cfg.get("max_rounds", 6))
        at_all_rounds = max_rounds // 2  # 前半段 @全体，后半段 @个人

        current_state = initial_state
        round_num = 1

        while round_num <= max_rounds and not current_state.get("done"):
            # 检查是否超时
            now_dt = _now()
            try:
                from datetime import datetime
                dl = datetime.fromisoformat(current_state["deadline_time"])
                if dl.tzinfo is None:
                    dl = dl.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                dl = now_dt

            if now_dt >= dl:
                # 超时
                _log("Deadline reached, ending session.")
                timeout_state = _report_timeout(current_state)
                report = timeout_state.get("report_text", "")
                if report:
                    admin_qq = str(cfg.get("admin_qq") or QQnumber or "").strip()
                    if admin_qq:
                        await _send_private_msg(admin_qq, report + "\n\n— [报平安]")
                        _log(f"Timeout report sent to admin {admin_qq}")
                _end_session(group_id, "timeout", report, round_num)
                return

            # ── 群内提醒 ─────────────────────────────────────────
            escalation = current_state.get("escalation_type", "at_all")
            pending = current_state.get("pending", [])
            if escalation in ("at_all", "at_person") and pending:
                try:
                    if escalation == "at_all":
                        at_all_msg = MessageArray()
                        at_all_msg.add_at_all()
                        await asyncio.wait_for(bot.api.post_group_array_msg(group_id, at_all_msg), timeout=10.0)
                        await asyncio.wait_for(bot.api.post_group_msg(group_id, text=f"报平安接龙（第{round_num}轮）：请回复「1」完成接龙"), timeout=10.0)
                    else:
                        names = [target_qqs.get(qq, qq) for qq in pending]
                        at_persons_msg = MessageArray()
                        for qq in pending:
                            at_persons_msg.add_at(qq)
                        await asyncio.wait_for(bot.api.post_group_array_msg(group_id, at_persons_msg), timeout=10.0)
                        await asyncio.wait_for(bot.api.post_group_msg(group_id, text=f"请以上同学回复「1」完成接龙：{', '.join(names)}"), timeout=10.0)
                    _log(f"Round {round_num} reminder sent ({escalation}) to group {group_id}")
                except Exception as e:
                    _log(f"[WARN] Reminder failed ({e}), sending text fallback")
                    try:
                        await asyncio.wait_for(bot.api.post_group_msg(group_id, text=f"【报平安提醒】请回复「1」完成接龙（第{round_num}轮）"), timeout=10.0)
                    except Exception:
                        pass

            # ── 等待下一轮 ─────────────────────────────────────
            _log(f"Round {round_num}: waiting {interval_seconds}s... pending={len(pending)}")
            await asyncio.sleep(interval_seconds)

            # ── 检查新消息，更新 pending ─────────────────────────
            msg_state = await _check_messages(current_state)
            pending = msg_state.get("pending", [])
            checked = msg_state.get("checked_in", [])
            _log(f"After check: checked={len(checked)}, pending={len(pending)}")

            # ── 全员已完成，提前结束 ─────────────────────────────
            if not pending:
                current_state = {**current_state, "round": round_num, "done": True, "pending": [], "checked_in": checked}
                break

            # ── 计算下一轮 ──────────────────────────────────────
            round_num += 1
            if round_num > max_rounds:
                _log(f"Max rounds ({max_rounds}) reached, exiting cycle")
                current_state = {**current_state, "round": max_rounds, "done": False, "pending": pending, "checked_in": checked}
                break

            next_escalation = "at_all" if round_num <= at_all_rounds else "at_person"
            current_state = _update_state_for_next_round({
                **current_state,
                "round": round_num,
                "checked_in": checked,
                "pending": pending,
                "last_check_time": _now().isoformat(timespec="seconds"),
                "escalation_type": next_escalation,
            })
            _log(f"Next round: {round_num}, escalation={next_escalation}, pending={len(pending)}")

        # ── 循环结束：全员完成或已达3轮 ─────────────────────────
        if not current_state.get("pending"):
            report_state = _report_completed(current_state)
            report = report_state.get("report_text", "")
        else:
            report_state = _report_timeout(current_state)
            report = report_state.get("report_text", "")

        admin_qq = str(cfg.get("admin_qq") or QQnumber or "").strip()
        if report and admin_qq:
            await _send_private_msg(admin_qq, report + "\n\n— [报平安]")
            _log(f"Final report sent to admin {admin_qq}")
        _end_session(group_id, "completed" if current_state.get("done") else "timeout", report, round_num)


# ─────────────────────────────────────────────
# 单例
# ─────────────────────────────────────────────

safety_checkin_scheduler = SafetyCheckinScheduler()


# ─────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────

def handle_group_message_for_checkin(group_id: str, user_id: str, message: str) -> bool:
    """
    在 group_entrance 中调用，严格匹配关键词，写入数据库。
    返回是否识别为有效接龙。
    """
    keyword = SAFETY_CONFIG.get("checkin_keyword", "1")
    cleaned = message.strip()
    if cleaned != keyword:
        return False

    groups = _monitor_groups()
    if group_id not in groups:
        return False

    target_qqs = _members()
    if user_id not in target_qqs:
        return False

    # 获取当前轮次
    today = _today()
    with _db_connect() as conn:
        row = conn.execute("""
            SELECT MAX(final_round) as max_round FROM checkin_session
            WHERE date=? AND group_id=? AND status='active'
        """, (today, group_id)).fetchone()
        current_round = int(row["max_round"]) if row and row["max_round"] else 1

    marked = _mark_checked_in(user_id, group_id, current_round)
    if marked:
        _log(f"Check-in recorded: {target_qqs.get(user_id, user_id)} ({user_id})")
    return marked
