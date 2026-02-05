"""Agent 管理器 - 协调多个 Agent 的消息处理"""
import asyncio
import time
from typing import Dict, Any, List
from loguru import logger
from datetime import datetime


class AgentManager:
    """Agent 管理器 - 负责 Agent 注册、消息路由和队列处理"""

    def __init__(self, napcat_client):
        """
        初始化 Agent 管理器

        Args:
            napcat_client: NapCat 客户端实例
        """
        self.napcat_client = napcat_client
        self.agents: Dict[str, Any] = {}  # agent_id -> agent 实例
        self.queues: Dict[str, asyncio.Queue] = {}  # agent_id -> 消息队列
        self.workers: Dict[str, asyncio.Task] = {}  # agent_id -> worker 任务

        logger.info("AgentManager 初始化完成")

    def register_agent(self, agent_id: str, agent):
        """
        注册 Agent

        Args:
            agent_id: Agent ID
            agent: Agent 实例
        """
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} 已存在，将被覆盖")

        self.agents[agent_id] = agent
        self.queues[agent_id] = asyncio.Queue()

        logger.info(f"Agent 已注册: {agent_id} ({agent.agent_name})")

    def start_workers(self):
        """启动所有 Agent 的 worker 任务"""
        for agent_id, agent in self.agents.items():
            if agent_id not in self.workers:
                worker = asyncio.create_task(self._agent_worker(agent_id))
                self.workers[agent_id] = worker
                logger.info(f"Agent worker 已启动: {agent_id}")

    async def _agent_worker(self, agent_id: str):
        """
        Agent worker 任务 - 处理队列中的消息

        Args:
            agent_id: Agent ID
        """
        agent = self.agents[agent_id]
        queue = self.queues[agent_id]

        logger.info(f"Agent worker 运行中: {agent_id}")

        while True:
            try:
                # 从队列获取消息
                message_data = await queue.get()

                # 检查 Agent 是否被禁用
                if hasattr(agent, 'enabled') and not agent.enabled:
                    logger.debug(f"[{agent.agent_name}] 已禁用，跳过消息处理")
                    queue.task_done()
                    continue

                # 记录开始时间
                start_time = time.time()

                # 调用 Agent 处理消息
                logger.debug(f"[{agent.agent_name}] 开始处理消息")
                result = await agent.process_message(message_data)

                # 计算处理时长
                processing_time = time.time() - start_time

                # 更新统计信息
                if hasattr(agent, 'stats'):
                    agent.stats["total_processed"] = agent.stats.get("total_processed", 0) + 1
                    agent.stats["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    agent.stats["total_time"] = agent.stats.get("total_time", 0.0) + processing_time
                    logger.debug(f"[{agent.agent_name}] 统计已更新: total_processed={agent.stats['total_processed']}, total_time={agent.stats['total_time']:.2f}s")

                # 记录结果
                if result:
                    logger.success(f"[{agent.agent_name}] 处理完成，耗时: {processing_time:.2f}s")
                else:
                    logger.debug(f"[{agent.agent_name}] 处理完成（无响应），耗时: {processing_time:.2f}s")

                # 标记任务完成
                queue.task_done()

            except Exception as e:
                logger.error(f"Agent worker 异常 ({agent_id}): {e}", exc_info=True)
                queue.task_done()

    async def route_message(self, message_data: Dict[str, Any]):
        """
        路由消息到合适的 Agent

        Args:
            message_data: NapCat 消息数据
        """
        try:
            # 收集所有触发的 Agent
            triggered_agents = []

            for agent_id, agent in self.agents.items():
                # 检查 Agent 是否启用
                if hasattr(agent, 'enabled') and not agent.enabled:
                    continue

                # 检查是否应该触发
                if agent.should_trigger(message_data):
                    triggered_agents.append(agent_id)

            # 如果没有 Agent 触发，跳过
            if not triggered_agents:
                logger.debug("没有 Agent 触发，跳过消息")
                return

            # 记录触发的 Agent
            agent_names = [self.agents[aid].agent_name for aid in triggered_agents]
            logger.info(f"触发 Agent: {', '.join(agent_names)}")

            # 将消息加入所有触发的 Agent 的队列（并行处理）
            for agent_id in triggered_agents:
                await self.queues[agent_id].put(message_data)
                logger.debug(f"消息已加入队列: {agent_id}")

        except Exception as e:
            logger.error(f"路由消息异常: {e}", exc_info=True)

    def get_agent(self, agent_id: str):
        """
        获取 Agent 实例

        Args:
            agent_id: Agent ID

        Returns:
            Agent 实例
        """
        return self.agents.get(agent_id)

    def get_all_agents(self) -> Dict[str, Any]:
        """
        获取所有 Agent

        Returns:
            Agent 字典
        """
        return self.agents

    def enable_agent(self, agent_id: str):
        """
        启用 Agent

        Args:
            agent_id: Agent ID
        """
        agent = self.agents.get(agent_id)
        if agent and hasattr(agent, 'enabled'):
            agent.enabled = True
            logger.info(f"Agent 已启用: {agent_id}")
        else:
            logger.warning(f"Agent 不存在或不支持启用/禁用: {agent_id}")

    def disable_agent(self, agent_id: str):
        """
        禁用 Agent

        Args:
            agent_id: Agent ID
        """
        agent = self.agents.get(agent_id)
        if agent and hasattr(agent, 'enabled'):
            agent.enabled = False
            logger.info(f"Agent 已禁用: {agent_id}")
        else:
            logger.warning(f"Agent 不存在或不支持启用/禁用: {agent_id}")

    def get_stats(self) -> List[Dict[str, Any]]:
        """
        获取所有 Agent 的统计信息

        Returns:
            统计信息列表
        """
        stats = []
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'get_stats'):
                agent_stats = agent.get_stats()
                agent_stats["enabled"] = getattr(agent, 'enabled', True)
                stats.append(agent_stats)
            else:
                stats.append({
                    "agent_id": agent_id,
                    "agent_name": getattr(agent, 'agent_name', agent_id),
                    "enabled": getattr(agent, 'enabled', True)
                })
        return stats

    async def shutdown(self):
        """关闭所有 worker 任务"""
        logger.info("正在关闭 AgentManager...")

        # 取消所有 worker 任务
        for agent_id, worker in self.workers.items():
            worker.cancel()
            logger.info(f"Agent worker 已取消: {agent_id}")

        # 等待所有队列清空
        for agent_id, queue in self.queues.items():
            await queue.join()
            logger.info(f"Agent 队列已清空: {agent_id}")

        logger.info("AgentManager 已关闭")
