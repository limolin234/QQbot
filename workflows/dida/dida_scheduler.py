import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ncatbot.core import GroupMessage, PrivateMessage

from bot import bot
from ..agent_config_loader import (
    is_scheduler_enabled,
    load_agent_config_by_filename,
    load_current_agent_config,
)
from .dida_service import DidaService
from ..scheduler.registry import action_registry


def _now() -> datetime:
    return datetime.now().astimezone()


def _normalize_timezone_offset(value: str) -> str:
    match = re.search(r"([+-])(\d{2})(\d{2})$", value)
    if not match:
        return value
    sign, hours, minutes = match.groups()
    return value[: match.start()] + f"{sign}{hours}:{minutes}"


def _parse_dida_datetime(text: str) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    normalized = _normalize_timezone_offset(normalized)
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _format_due_date(value: str) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    normalized = _normalize_timezone_offset(normalized)
    dt = None
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        pass
    if dt is None:
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d %H:%M")
        except ValueError:
            return None
        dt = dt.replace(tzinfo=_now().tzinfo)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_now().tzinfo)
    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def _route_tag(chat_type: str, target_id: str, creator_id: str) -> str:
    return f"[qq:{chat_type}:{target_id}][creator:{creator_id}]"


def _extract_route(text: str) -> tuple[str, str, str] | None:
    content = str(text or "")
    match = re.search(r"\[qq:(group|private):(\d+)\]", content)
    if not match:
        return None
    chat_type = match.group(1)
    target_id = match.group(2)
    creator_match = re.search(r"\[creator:(\d+)\]", content)
    creator_id = creator_match.group(1) if creator_match else ""
    return chat_type, target_id, creator_id


class DidaScheduler:
    def __init__(self, *, token_path: str) -> None:
        self.token_path = token_path
        self.notified: dict[str, float] = {}
        self.lock = asyncio.Lock()

    def _log(self, message: str) -> None:
        print(f"[DIDA] {message}")

    def _load_config(self) -> dict[str, Any]:
        return load_current_agent_config(__file__)

    def _load_dida_agent_config(self) -> dict[str, Any]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "agent_config.yaml")
        return load_agent_config_by_filename("dida_agent.py", config_path=config_path)

    @staticmethod
    def _normalize_qq_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return []

    def _get_service(self) -> DidaService | None:
        config = self._load_config()
        client_id = str(config.get("client_id") or "").strip()
        client_secret = str(config.get("client_secret") or "").strip()
        redirect_uri = str(config.get("redirect_uri") or "").strip()
        if not client_id or not client_secret or not redirect_uri:
            return None
        return DidaService(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)

    def _get_runtime_config(self) -> dict[str, Any]:
        config = self._load_config()
        dida_agent_config = self._load_dida_agent_config()
        poll_interval_seconds = int(config.get("poll_interval_seconds", 60))
        due_window_seconds = int(config.get("due_window_seconds", 60))
        max_tasks_scan = int(config.get("max_tasks_scan_per_user", 200))
        project_ids = config.get("project_ids")
        if not isinstance(project_ids, list):
            project_ids = []
        admin_qqs = self._normalize_qq_list(dida_agent_config.get("admin_qqs", []))
        if not admin_qqs:
            admin_qqs = self._normalize_qq_list(config.get("admin_qqs", []))
            
        return {
            "poll_interval_seconds": max(poll_interval_seconds, 5),
            "due_window_seconds": max(due_window_seconds, 30),
            "max_tasks_scan_per_user": max(max_tasks_scan, 50),
            "project_ids": [str(item).strip() for item in project_ids if str(item).strip()],
            "admin_qqs": admin_qqs,
        }

    def load_tokens(self) -> dict[str, Any]:
        if not os.path.exists(self.token_path):
            return {}
        try:
            with open(self.token_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_tokens(self, payload: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
        with open(self.token_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

    async def handle_command(self, msg: GroupMessage | PrivateMessage) -> bool:
        text = str(getattr(msg, "raw_message", "") or "").strip()
        if not text:
            return False
        if not (text.startswith("/dida_auth") or text.startswith("/bind_dida")):
            return False
        service = self._get_service()
        if service is None:
            self._log("config_missing")
            await self._send_text(msg, "⚠️ Dida 未配置，请开发者先在 agent_config.yaml 中填写 client_id/client_secret/redirect_uri。")
            return True
        user_id = str(getattr(msg, "user_id", "") or "").strip()
        if text.startswith("/dida_auth"):
            self._log(f"oauth_start user={user_id}")
            state = f"qq_{user_id}_{int(_now().timestamp())}"
            url = service.get_oauth_url(state=state)
            await self._send_text(msg, f"请打开授权链接并复制 code：\n{url}\n完成后发送 /bind_dida code=xxxx")
            return True
        match = re.search(r"code\s*=\s*([^\s]+)", text)
        if not match:
            await self._send_text(msg, "请使用 /bind_dida code=xxxx 绑定 Dida 账号。")
            return True
        code = match.group(1).strip()
        self._log(f"oauth_exchange user={user_id}")
        token_data = await asyncio.to_thread(service.exchange_code, code=code)
        if not isinstance(token_data, dict) or not token_data.get("access_token"):
            self._log(f"oauth_failed user={user_id}")
            await self._send_text(msg, "⚠️ Dida 授权失败，请重试。")
            return True
        self._log(f"oauth_bound user={user_id}")
        tokens = self.load_tokens()
        tokens[user_id] = {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "bound_at": _now().isoformat(),
        }
        self.save_tokens(tokens)
        project_id = await self._ensure_default_project_id(user_id, tokens[user_id], service)
        if project_id:
            self.save_tokens(tokens)
        await self._send_text(msg, "✅ Dida 绑定成功，可以开始创建任务。")
        return True

    async def _find_task_obj(
        self,
        access_token: str,
        service: DidaService,
        *,
        task_id: str,
        project_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if project_id:
            target_project_ids = [project_id]
        else:
            projects = await asyncio.to_thread(service.get_projects, access_token=access_token)
            target_project_ids = []
            if isinstance(projects, list):
                target_project_ids = [str(p.get("id") or "").strip() for p in projects if p.get("id")]
            if "inbox" not in target_project_ids:
                target_project_ids.append("inbox")

        tasks_list = await asyncio.gather(
            *[
                asyncio.to_thread(service.get_project_data, access_token=access_token, project_id=pid)
                for pid in target_project_ids
            ],
            return_exceptions=True
        )

        for i, res in enumerate(tasks_list):
            if not isinstance(res, dict):
                continue
            pid = target_project_ids[i]
            project_tasks = res.get("tasks", [])
            if not isinstance(project_tasks, list):
                continue
            
            for task in project_tasks:
                tid = str(task.get("id", "") or "").strip()
                
                if tid == task_id:
                    return task, pid
                    
        return None, None

    def _find_project_id_from_context(self, user_id: str, task_id: str) -> str | None:
        try:
            path = os.path.join("data", "dida_context.json")
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            user_tasks = data.get(user_id, [])
            if not isinstance(user_tasks, list):
                return None
            
            for t in user_tasks:
                if str(t.get("id")) == task_id:
                    return t.get("projectId")
        except Exception:
            pass
        return None

    async def execute_action(
        self,
        *,
        action: Any,
        chat_type: str,
        group_id: str,
        user_id: str,
        user_name: str,
    ) -> str:
        """Compatible wrapper for old string-based return."""
        result = await self.execute_action_structured(
            action=action,
            chat_type=chat_type,
            group_id=group_id,
            user_id=user_id,
            user_name=user_name
        )
        return result.get("message", "")

    async def execute_action_structured(
        self,
        *,
        action: Any,
        chat_type: str,
        group_id: str,
        user_id: str,
        user_name: str,
    ) -> dict[str, Any]:
        service = self._get_service()
        if service is None:
            return {"ok": False, "message": "⚠️ Dida 未配置，请先在 agent_config.yaml 中填写 client_id/client_secret/redirect_uri。"}
        user_key = str(user_id).strip()
        
        # Admin impersonation check
        target_user_id = str(getattr(action, "target_user_id", "") or "").strip()
        if target_user_id and target_user_id != user_key:
            runtime_config = self._get_runtime_config()
            admin_qqs = runtime_config.get("admin_qqs", [])
            if user_key in admin_qqs:
                user_key = target_user_id
            else:
                return {"ok": False, "message": f"⚠️ 权限不足：你不是管理员，无法操作用户 {target_user_id} 的任务。"}
                
        tokens = self.load_tokens()
        token_data = tokens.get(user_key)
        if not isinstance(token_data, dict):
            return {"ok": False, "message": "⚠️ 尚未绑定 Dida 账号，请先发送 /dida_auth 获取授权链接。"}
        access_token = str(token_data.get("access_token") or "").strip()
        if not access_token:
            return {"ok": False, "message": "⚠️ Dida 授权信息无效，请重新绑定 /dida_auth。"}
        project_id = str(getattr(action, "project_id", "") or "").strip()
        
        # Try to find project_id from context if missing and we have a task_id
        task_id_for_lookup = str(getattr(action, "task_id", "") or "").strip()
        if not project_id and task_id_for_lookup:
            found_pid = self._find_project_id_from_context(user_key, task_id_for_lookup)
            if found_pid:
                project_id = found_pid
        
        action_type = str(getattr(action, "action_type", "") or "").strip()
        self._log(f"action_start type={action_type or 'unknown'} user={user_key} project={project_id or 'default'}")
        
        if action_type == "create":
            if not project_id:
                project_id = await self._ensure_default_project_id(user_key, token_data, service)
                if project_id:
                    self.save_tokens(tokens)
            
            title = str(getattr(action, "title", "") or "").strip()
            if not title:
                return {"ok": False, "message": "⚠️ 创建任务需要 title。"}
            if not project_id:
                return {"ok": False, "message": "⚠️ 未找到默认项目，请先在 Dida 中创建项目。"}
            content = str(getattr(action, "content", "") or "").strip()
            desc = str(getattr(action, "desc", "") or "").strip()
            due_date_raw = str(getattr(action, "due_date", "") or "").strip()
            due_date = _format_due_date(due_date_raw) if due_date_raw else None
            if due_date_raw and not due_date:
                return {"ok": False, "message": "⚠️ due_date 格式不正确，请提供明确时间。"}
            target_id = group_id if chat_type == "group" else user_id
            route = _route_tag(chat_type, str(target_id), user_id)
            content = f"{content}\n{route}".strip() if content else route
            payload: dict[str, Any] = {"title": title, "projectId": project_id, "content": content}
            if desc:
                payload["desc"] = desc
            if due_date:
                payload["dueDate"] = due_date
            is_all_day = getattr(action, "is_all_day", None)
            if isinstance(is_all_day, bool):
                payload["isAllDay"] = is_all_day
            time_zone = str(getattr(action, "time_zone", "") or "").strip()
            if time_zone:
                payload["timeZone"] = time_zone
            repeat_flag = str(getattr(action, "repeat_flag", "") or "").strip()
            if repeat_flag:
                payload["repeatFlag"] = repeat_flag
            reminders = getattr(action, "reminders", None)
            if isinstance(reminders, list) and reminders:
                payload["reminders"] = [str(item) for item in reminders if str(item).strip()]
            created = await asyncio.to_thread(service.create_task, access_token=access_token, payload=payload)
            task_id = str(created.get("id") or created.get("taskId") or "").strip()
            self._log(f"task_created user={user_key} project={project_id} task={task_id or 'unknown'}")

            due_display = ""
            if due_date:
                dt = _parse_dida_datetime(due_date)
                if dt:
                    dt_local = dt.astimezone()
                    if bool(payload.get("isAllDay")):
                        due_display = dt_local.strftime("%m-%d") + " (全天)"
                    else:
                        due_display = dt_local.strftime("%m-%d %H:%M")

            if due_display:
                return {"ok": True, "message": f"✅ 已创建任务：{title} 📅 {due_display}", "data": {"task_id": task_id, "title": title}}
            return {"ok": True, "message": f"✅ 已创建任务：{title}", "data": {"task_id": task_id, "title": title}}
        if action_type == "update":
            target_task_id = str(getattr(action, "task_id", "") or "").strip()
            title = str(getattr(action, "title", "") or "").strip()
            content = str(getattr(action, "content", "") or "").strip()
            due_date_raw = str(getattr(action, "due_date", "") or "").strip()
            due_date = _format_due_date(due_date_raw) if due_date_raw else None

            if not target_task_id:
                 return {"ok": False, "message": "⚠️ 更新任务必须提供 task_id。"}

            task_obj, pid = await self._find_task_obj(
                access_token, 
                service, 
                task_id=target_task_id, 
                project_id=project_id
            )
            
            if not task_obj or not pid:
                return {"ok": False, "message": "⚠️ 未找到目标任务，请检查任务 ID。"}
            
            payload = task_obj.copy()
            if title:
                payload["title"] = title
            if content:
                payload["content"] = content
            if due_date:
                payload["dueDate"] = due_date
            
            payload["id"] = task_obj["id"]
            payload["projectId"] = pid
            
            await asyncio.to_thread(
                service.update_task, 
                access_token=access_token, 
                task_id=payload["id"], 
                payload=payload
            )
            
            self._log(f"task_updated user={user_key} project={pid} task={payload['id']}")

            due_display = ""
            due_value = str(payload.get("dueDate") or "").strip()
            if due_value:
                dt = _parse_dida_datetime(due_value)
                if dt:
                    dt_local = dt.astimezone()
                    if bool(payload.get("isAllDay")):
                        due_display = dt_local.strftime("%m-%d") + " (全天)"
                    else:
                        due_display = dt_local.strftime("%m-%d %H:%M")

            if due_display:
                return {"ok": True, "message": f"✅ 已更新任务：{payload.get('title')} 📅 {due_display}", "data": {"task_id": payload["id"]}}
            return {"ok": True, "message": f"✅ 已更新任务：{payload.get('title')}", "data": {"task_id": payload["id"]}}
        if action_type == "list":
            # 如果用户未显式指定 project_id，则忽略默认项目，拉取所有项目
            raw_project_id = str(getattr(action, "project_id", "") or "").strip()
            target_project_ids = []
            if raw_project_id:
                target_project_ids = [raw_project_id]
            else:
                projects = await asyncio.to_thread(service.get_projects, access_token=access_token)
                if isinstance(projects, list):
                    target_project_ids = [str(p.get("id") or "").strip() for p in projects if p.get("id")]
                # Explicitly add inbox
                if "inbox" not in target_project_ids:
                    target_project_ids.append("inbox")
            
            if not target_project_ids:
                return {"ok": False, "message": "⚠️ 未找到任何项目，请先在 Dida 中创建项目。"}

            self._log(f"fetch_tasks projects={len(target_project_ids)}")
            tasks_list = await asyncio.gather(
                *[
                    asyncio.to_thread(service.get_project_data, access_token=access_token, project_id=pid)
                    for pid in target_project_ids
                ],
                return_exceptions=True
            )

            project_groups = []
            total_tasks_count = 0

            for i, res in enumerate(tasks_list):
                if not isinstance(res, dict):
                    continue
                
                tasks = res.get("tasks", [])
                if not isinstance(tasks, list) or not tasks:
                    continue
                
                active_tasks = [t for t in tasks if str(t.get("status", "0")) == "0"]
                if not active_tasks:
                    continue

                project_info = res.get("project", {})
                p_name = str(project_info.get("name", "") or "").strip()
                if not p_name:
                    pid = target_project_ids[i]
                    p_name = "收集箱" if pid == "inbox" else "未命名项目"
                
                project_groups.append((p_name, active_tasks))
                total_tasks_count += len(active_tasks)
            
            # Update context for AI immediately after list
            all_tasks_for_context = []
            
            # Group tasks for better display
            now = _now()
            today_str = now.strftime("%Y-%m-%d")
            tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            
            groups: dict[str, list[tuple[str, dict[str, Any]]]] = {
                "已过期": [],
                "今天": [],
                "明天": [],
                "以后": [],
                "无日期": []
            }

            for p_name, p_tasks in project_groups:
                for t in p_tasks:
                    all_tasks_for_context.append({
                        "id": t.get("id"),
                        "title": t.get("title"),
                        "project": p_name,
                        "projectId": t.get("projectId"),
                        "due": t.get("dueDate"),
                        "isAllDay": t.get("isAllDay")
                    })
                    
                    due_raw = str(t.get("dueDate", "") or "").strip()
                    if not due_raw:
                        groups["无日期"].append((p_name, t))
                        continue
                        
                    dt = _parse_dida_datetime(due_raw)
                    if not dt:
                        groups["无日期"].append((p_name, t))
                        continue
                        
                    dt_local = dt.astimezone()
                    dt_str = dt_local.strftime("%Y-%m-%d")
                    
                    if dt_local < now and dt_str != today_str:
                        groups["已过期"].append((p_name, t))
                    elif dt_str == today_str:
                        groups["今天"].append((p_name, t))
                    elif dt_str == tomorrow_str:
                        groups["明天"].append((p_name, t))
                    else:
                        groups["以后"].append((p_name, t))

            self._save_task_context(user_key, all_tasks_for_context)
            
            self._log(f"fetch_tasks_done total={total_tasks_count}")

            if not total_tasks_count:
                return {"ok": True, "message": f"[CQ:at,qq={user_key}] 暂无未完成任务。", "data": {"count": 0}}

            output_parts = [f"[CQ:at,qq={user_key}] 的任务清单 (共 {total_tasks_count} 项)"]
            output_parts.append("-" * 15)

            for group_name in ["已过期", "今天", "明天", "以后", "无日期"]:
                task_items = groups[group_name]
                if not task_items:
                    continue
                
                # Sort within group by due date
                task_items.sort(key=lambda x: str(x[1].get("dueDate", "")))
                
                output_parts.append(f"[{group_name}]")
                for p_name, task in task_items:
                    title = str(task.get("title", "") or "").strip()
                    due_raw = str(task.get("dueDate", "") or "").strip()
                    
                    due_info = ""
                    if due_raw:
                        dt = _parse_dida_datetime(due_raw)
                        if dt:
                            dt_local = dt.astimezone()
                            if bool(task.get("isAllDay")):
                                due_info = f" {dt_local.strftime('%m-%d')}"
                            else:
                                due_info = f" {dt_local.strftime('%m-%d %H:%M')}"
                    
                    line = f"{title}"
                    if due_info:
                        line += f" {due_info}"
                    output_parts.append(line)
                output_parts.append("") # Empty line between groups

            return {"ok": True, "message": "\n".join(output_parts).strip(), "data": {"count": total_tasks_count}}

        if action_type in {"delete", "complete"}:
            task_id = str(getattr(action, "task_id", "") or "").strip()
            
            if not task_id:
                return {"ok": False, "message": "⚠️ 操作任务必须提供 task_id。"}

            task_obj, pid = await self._find_task_obj(
                access_token, 
                service, 
                task_id=task_id, 
                project_id=project_id
            )
            
            if not task_obj or not pid:
                return {"ok": False, "message": "⚠️ 未找到目标任务，请检查 task_id。"}
            
            target_task_id = task_obj["id"]
            target_project_id = str(task_obj.get("projectId") or pid).strip()

            if action_type == "delete":
                await asyncio.to_thread(service.delete_task, access_token=access_token, project_id=target_project_id, task_id=target_task_id)
                self._log(f"task_deleted user={user_key} project={target_project_id} task={target_task_id}")
                return {"ok": True, "message": f"✅ 已删除任务：{task_obj.get('title', 'unknown')}", "data": {"task_id": target_task_id}}
            
            await asyncio.to_thread(service.complete_task, access_token=access_token, project_id=target_project_id, task_id=target_task_id)
            self._log(f"task_completed user={user_key} project={target_project_id} task={target_task_id}")
            return {"ok": True, "message": f"✅ 已完成任务：{task_obj.get('title', 'unknown')}", "data": {"task_id": target_task_id}}
        return {"ok": False, "message": "⚠️ 未识别的 Dida 操作。"}

    async def start(self) -> None:
        if is_scheduler_enabled():
            print("[DIDA] Scheduler enabled, skipping internal loop.")
            # Register actions
            action_registry.register("dida.poll", self.poll_action)
            action_registry.register("dida.push_task_list", self.push_task_list_action)
            return

        print("[DIDA] Scheduler disabled (legacy mode), starting internal loop...")
        while True:
            config = self._get_runtime_config()
            try:
                await self.poll_once(config)
            except Exception as error:
                print(f"[DidaScheduler] error={error}")
            await asyncio.sleep(config["poll_interval_seconds"])

    async def poll_action(self, project_ids: list[str] = None) -> None:
        """Wrapper for scheduler to call poll_once with overridden config."""
        config = self._get_runtime_config()
        if project_ids:
            # Override project_ids if provided
            config["project_ids"] = [str(pid).strip() for pid in project_ids]
        await self.poll_once(config)

    async def push_task_list_action(
        self, 
        group_id: str = None, 
        user_qq: str = None, 
        day_range: str = "today", 
        limit: int = 10
    ) -> None:
        """Action: Push task list to a group or private user."""
        service = self._get_service()
        if not service:
            self._log("push_task_list skipped: service not configured")
            return
            
        target_user_qq = str(user_qq or "").strip()
        if not target_user_qq:
             self._log("push_task_list skipped: user_qq is required")
             return

        tokens = self.load_tokens()
        token_data = tokens.get(target_user_qq)
        if not token_data or not token_data.get("access_token"):
            self._log(f"push_task_list skipped: user {target_user_qq} not bound")
            return

        access_token = token_data.get("access_token")
        
        # 1. Fetch all tasks
        projects = await asyncio.to_thread(service.get_projects, access_token=access_token)
        target_project_ids = []
        if isinstance(projects, list):
            target_project_ids = [str(p.get("id")) for p in projects if p.get("id")]
        if "inbox" not in target_project_ids:
            target_project_ids.append("inbox")
            
        all_tasks = []
        tasks_list = await asyncio.gather(
            *[
                asyncio.to_thread(service.get_project_data, access_token=access_token, project_id=pid)
                for pid in target_project_ids
            ],
            return_exceptions=True
        )
        
        for res in tasks_list:
            if isinstance(res, dict):
                tasks = res.get("tasks", [])
                if isinstance(tasks, list):
                    all_tasks.extend(tasks)

        # 2. Filter tasks
        filtered_tasks = []
        now = _now()
        today_str = now.strftime("%Y-%m-%d")
        tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        for task in all_tasks:
            status = str(task.get("status", "0"))
            if status != "0":  # Skip completed
                continue
                
            due_raw = task.get("dueDate")
            if not due_raw:
                # If day_range is 'all', include no-date tasks? 
                # For now, let's say 'all' includes everything, but 'today'/'overdue' requires date.
                if day_range == "all":
                    filtered_tasks.append(task)
                continue
                
            dt = _parse_dida_datetime(due_raw)
            if not dt:
                continue
                
            dt_local = dt.astimezone()
            dt_str = dt_local.strftime("%Y-%m-%d")
            
            is_overdue = dt_local < now and dt_str != today_str
            is_today = dt_str == today_str
            is_tomorrow = dt_str == tomorrow_str
            
            if day_range == "overdue" and is_overdue:
                filtered_tasks.append(task)
            elif day_range == "today" and (is_overdue or is_today):
                filtered_tasks.append(task)
            elif day_range == "tomorrow" and is_tomorrow:
                filtered_tasks.append(task)
            elif day_range == "all":
                filtered_tasks.append(task)

        # 3. Sort and Limit
        # Sort by due date
        filtered_tasks.sort(key=lambda x: str(x.get("dueDate", "") or "z"))
        
        if limit > 0:
            filtered_tasks = filtered_tasks[:limit]
            
        if not filtered_tasks:
            self._log(f"push_task_list: no tasks found for {target_user_qq} range={day_range}")
            # Optionally send "No tasks" message?
            return

        # 4. Format Message
        lines = [f"📋 待办清单 ({day_range})"]
        for task in filtered_tasks:
            title = str(task.get("title", "")).strip()
            due_raw = task.get("dueDate")
            due_info = ""
            if due_raw:
                dt = _parse_dida_datetime(due_raw)
                if dt:
                    if bool(task.get("isAllDay")):
                        due_info = f" [{dt.astimezone().strftime('%m-%d')}]"
                    else:
                        due_info = f" [{dt.astimezone().strftime('%H:%M')}]"
            
            lines.append(f"- {title}{due_info}")
            
        message = "\n".join(lines)
        
        # 5. Send Message
        if group_id:
            await bot.api.post_group_msg(str(group_id), text=f"[CQ:at,qq={target_user_qq}]\n{message}")
        else:
            await bot.api.post_private_msg(str(target_user_qq), text=message)
            
        self._log(f"push_task_list sent to {target_user_qq} (group={group_id}) count={len(filtered_tasks)}")

    async def poll_once(self, config: dict[str, Any]) -> None:
        service = self._get_service()
        if service is None:
            return
        tokens = self.load_tokens()
        self._log(f"poll_once users={len(tokens)} projects_config={len(config.get('project_ids', []))}")
        for user_id, token_data in tokens.items():
            access_token = str(token_data.get("access_token") or "").strip()
            if not access_token:
                continue
            project_ids = list(config["project_ids"])
            if not project_ids:
                try:
                    projects = await asyncio.to_thread(service.get_projects, access_token=access_token)
                    if isinstance(projects, list):
                        project_summary = [f"{p.get('name')}:{p.get('id')}" for p in projects]
                        self._log(f"poll_projects_fetched user={user_id} projects={project_summary}")
                        project_ids = [str(p.get("id") or "").strip() for p in projects if p.get("id")]
                        # Explicitly add inbox as it is not returned by get_projects
                        if "inbox" not in project_ids:
                            project_ids.append("inbox")
                except Exception as e:
                    self._log(f"poll_projects_failed user={user_id} error={e}")
                    continue
            
            if not project_ids:
                continue
            
            # Context for AI
            user_active_tasks = []

            for project_id in project_ids:
                data = await asyncio.to_thread(service.get_project_data, access_token=access_token, project_id=project_id)
                tasks = data.get("tasks", []) if isinstance(data, dict) else []
                project_info = data.get("project", {}) if isinstance(data, dict) else {}
                project_name = project_info.get("name", "unknown")
                if project_id == "inbox" and project_name == "unknown":
                    project_name = "Inbox"
                self._log(f"poll_project user={user_id} project={project_name}({project_id}) tasks={len(tasks)}")
                
                # Collect active tasks for AI context
                for task in tasks:
                     if str(task.get("status", "0")) == "0":
                         user_active_tasks.append({
                             "id": task.get("id"),
                             "title": task.get("title"),
                             "project": project_name,
                             "projectId": project_id,
                             "due": task.get("dueDate"),
                             "isAllDay": task.get("isAllDay")
                         })

                if not tasks and isinstance(data, dict):
                     # Debug: print raw data keys to ensure we are parsing correctly
                     self._log(f"poll_project_empty_detail keys={list(data.keys())} project_meta={project_info}")
                
                scanned = 0
                for task in tasks:
                    if scanned >= config["max_tasks_scan_per_user"]:
                        break
                    scanned += 1
                    if str(task.get("status", "0")) != "0":
                        continue
                    due_dt = _parse_dida_datetime(task.get("dueDate"))
                    if due_dt is None:
                        continue
                    now = _now()
                    if not (now <= due_dt <= now + timedelta(seconds=config["due_window_seconds"])):
                        continue
                    route = _extract_route(task.get("content")) or _extract_route(task.get("desc"))
                    if not route:
                        continue
                    chat_type, target_id, _ = route
                    key = f"{user_id}:{task.get('id')}:{task.get('dueDate')}"
                    if not self._should_notify(key):
                        continue
                    title = str(task.get("title", "") or "").strip()
                    due_display = due_dt.astimezone().strftime("%Y-%m-%d %H:%M")
                    if bool(task.get("isAllDay")):
                        due_display = due_dt.astimezone().strftime("%Y-%m-%d") + " (全天)"
                    message = f"⏰ 任务到期提醒\n标题：{title}\n到期：{due_display}\nID: {task.get('id')}"
                    # Use user_id (the task owner) as the mention target instead of creator_id
                    await self._send_route(chat_type, target_id, user_id, message)
            
            # Save context for AI
            self._save_task_context(user_id, user_active_tasks)
        self._gc_notified()

    def _save_task_context(self, user_id: str, tasks: list[dict[str, Any]]) -> None:
        try:
            path = os.path.join("data", "dida_context.json")
            # Simple implementation: Overwrite with latest user data (assuming single user mainly or merging)
            # For multi-user, we might want a dict keyed by user_id
            
            current_data = {}
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        current_data = json.load(f)
                except:
                    pass
            
            current_data[user_id] = tasks
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"save_context_failed error={e}")

    async def _send_route(self, chat_type: str, target_id: str, mention_user_id: str, text: str) -> None:
        if chat_type == "group":
            prefix = f"[CQ:at,qq={mention_user_id}] " if mention_user_id else ""
            await bot.api.post_group_msg(str(target_id), text=prefix + text)
        else:
            await bot.api.post_private_msg(str(target_id), text=text)

    async def _send_text(self, msg: GroupMessage | PrivateMessage, text: str) -> None:
        if isinstance(msg, GroupMessage):
            await bot.api.post_group_msg(str(msg.group_id), text=text)
        else:
            await bot.api.post_private_msg(str(msg.user_id), text=text)

    def _should_notify(self, key: str) -> bool:
        if key in self.notified:
            return False
        self.notified[key] = _now().timestamp()
        return True

    def _gc_notified(self) -> None:
        cutoff = _now().timestamp() - 86400
        self.notified = {key: ts for key, ts in self.notified.items() if ts >= cutoff}

    async def _ensure_default_project_id(
        self,
        user_id: str,
        token_data: dict[str, Any],
        service: DidaService,
    ) -> str:
        project_id = str(token_data.get("default_project_id") or "").strip()
        if project_id:
            return project_id
        access_token = str(token_data.get("access_token") or "").strip()
        if not access_token:
            return ""
        projects = await asyncio.to_thread(service.get_projects, access_token=access_token)
        if isinstance(projects, list) and projects:
            project_id = str(projects[0].get("id") or "").strip()
            if project_id:
                token_data["default_project_id"] = project_id
                return project_id
        return ""

    async def _find_task_by_id(
        self,
        access_token: str,
        service: DidaService,
        task_id: str,
        project_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """
        Find a task by ID across projects (including inbox).
        """
        if not task_id:
            return None, ""

        target_projects = []
        if project_id:
            target_projects = [project_id]
        else:
            projects = await asyncio.to_thread(service.get_projects, access_token=access_token)
            if isinstance(projects, list):
                target_projects = [str(p.get("id")) for p in projects if p.get("id")]
            if "inbox" not in target_projects:
                target_projects.append("inbox")
        
        for pid in target_projects:
            data = await asyncio.to_thread(service.get_project_data, access_token=access_token, project_id=pid)
            tasks = data.get("tasks", []) if isinstance(data, dict) else []
            for task in tasks:
                t_id = str(task.get("id") or "")
                if t_id == task_id:
                    return task, pid
        return None, ""

    async def _find_project_id_for_task(
        self,
        access_token: str,
        task_id: str,
        service: DidaService,
    ) -> str:
        projects = await asyncio.to_thread(service.get_projects, access_token=access_token)
        if not isinstance(projects, list):
            return ""
        for project in projects:
            project_id = str(project.get("id") or "").strip()
            if not project_id:
                continue
            data = await asyncio.to_thread(service.get_project_data, access_token=access_token, project_id=project_id)
            tasks = data.get("tasks", []) if isinstance(data, dict) else []
            for task in tasks:
                if str(task.get("id") or "") == task_id:
                    return project_id
        return ""


dida_scheduler = DidaScheduler(token_path=os.path.join("data", "dida_tokens.json"))
