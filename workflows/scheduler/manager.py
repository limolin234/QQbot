import asyncio
import aiocron
import logging
import os
import traceback
from typing import List
from zoneinfo import ZoneInfo
from .config_models import SchedulerConfig, ScheduleConfig, StepTreeNode
from .registry import action_registry
from ..agent_config_loader import load_agent_config_by_filename

logger = logging.getLogger(__name__)


def _normalize_cron_expression(raw: str) -> str:
    expression = (raw or "").strip()
    if not expression:
        return ""
    parts = expression.split()
    if len(parts) >= 6:
        # Config Studio may emit second-level input; aiocron runtime uses five-field expression.
        return " ".join(parts[1:6])
    return expression


class SchedulerManager:
    def __init__(self):
        self.cron_jobs = []
        self.interval_tasks = []
        self.config: SchedulerConfig | None = None

    def load_config(self) -> None:
        virtual_filename = "scheduler_manager.py"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "agent_config.yaml")
        raw_config = load_agent_config_by_filename(
            virtual_filename, config_path=config_path
        )

        if not raw_config:
            logger.warning(
                f"[Scheduler] No configuration found for '{virtual_filename}' in {config_path}. Skipping initialization."
            )
            return

        try:
            self.config = SchedulerConfig(**raw_config)
            logger.info(
                f"[Scheduler] Configuration loaded. Found {len(self.config.schedules)} schedules."
            )
        except Exception as e:
            logger.error(f"[Scheduler] Configuration validation failed: {e}")
            self.config = None

    async def start(self) -> None:
        self.load_config()
        if not self.config:
            return

        for schedule in self.config.schedules:
            if not schedule.enabled:
                continue

            if schedule.type == "cron":
                if not schedule.expression:
                    logger.error(
                        f"[Scheduler] Error: Cron schedule '{schedule.name}' missing expression."
                    )
                    continue
                self._schedule_cron(schedule)
            elif schedule.type == "interval":
                if not schedule.seconds:
                    logger.error(
                        f"[Scheduler] Error: Interval schedule '{schedule.name}' missing seconds."
                    )
                    continue
                self._schedule_interval(schedule)

    def _schedule_cron(self, schedule: ScheduleConfig) -> None:
        try:
            expression = _normalize_cron_expression(schedule.expression or "")
            if not expression:
                logger.error(
                    f"[Scheduler] Error: Cron schedule '{schedule.name}' expression is invalid."
                )
                return

            tz = None
            if self.config and self.config.timezone:
                try:
                    tz = ZoneInfo(self.config.timezone)
                except Exception:
                    logger.warning(
                        f"[Scheduler] Invalid timezone '{self.config.timezone}', falling back to system time."
                    )

            cron = aiocron.crontab(
                expression,
                func=self._make_job_func(schedule),
                start=True,
                tz=tz,
            )
            self.cron_jobs.append(cron)
            logger.info(
                f"[Scheduler] Scheduled cron job: {schedule.name} ({expression}) tz={self.config.timezone}"
            )
        except Exception as e:
            logger.error(
                f"[Scheduler] Failed to schedule cron job '{schedule.name}': {e}"
            )

    def _schedule_interval(self, schedule: ScheduleConfig) -> None:
        task = asyncio.create_task(self._interval_loop(schedule))
        self.interval_tasks.append(task)
        logger.info(
            f"[Scheduler] Scheduled interval job: {schedule.name} (every {schedule.seconds}s)"
        )

    async def _interval_loop(self, schedule: ScheduleConfig) -> None:
        while True:
            try:
                await asyncio.sleep(schedule.seconds)
                await self._run_job(schedule)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[Scheduler] Error in interval loop for '{schedule.name}': {e}"
                )
                logger.debug(traceback.format_exc())
                # Wait a bit before retrying to avoid tight loops on error
                await asyncio.sleep(5)

    def _make_job_func(self, schedule: ScheduleConfig):
        async def job_wrapper():
            await self._run_job(schedule)

        return job_wrapper

    async def _run_job(self, schedule: ScheduleConfig) -> None:
        logger.info(f"[Scheduler] Running job: {schedule.name}")
        if schedule.steps_tree:
            await self._execute_steps_tree(schedule.steps_tree, schedule.name)
            return

        for step in schedule.steps:
            try:
                action_func = action_registry.get(step.action)
                await action_func(**step.params)
            except Exception as e:
                logger.error(
                    f"[Scheduler] Job '{schedule.name}' step '{step.action}' failed: {e}"
                )
                logger.debug(traceback.format_exc())

    async def _execute_steps_tree(
        self,
        steps_tree: list[StepTreeNode],
        schedule_name: str,
    ) -> None:
        for node in steps_tree:
            await self._execute_step_node(node, schedule_name)

    async def _execute_step_node(
        self,
        node: StepTreeNode,
        schedule_name: str,
    ) -> None:
        try:
            if node.kind == "action":
                await self._execute_action_node(node, schedule_name)
                return

            if node.kind == "group":
                for child in node.children:
                    await self._execute_step_node(child, schedule_name)
                return

            logger.warning(
                f"[Scheduler] Unsupported step kind '{node.kind}' in schedule '{schedule_name}'"
            )
        except Exception as e:
            logger.error(
                f"[Scheduler] Schedule '{schedule_name}' step-tree node failed: {e}"
            )
            logger.debug(traceback.format_exc())

    async def _execute_action_node(
        self, node: StepTreeNode, schedule_name: str
    ) -> None:
        action = (node.action or "").strip()
        if not action:
            logger.warning(f"[Scheduler] Empty action in schedule '{schedule_name}'")
            return

        params = node.params if isinstance(node.params, dict) else {}
        action_func = action_registry.get(action)
        await action_func(**params)
