"""Agent 工厂 - 自动加载和创建 Agent 实例"""
from typing import Dict, Any
from loguru import logger


class AgentFactory:
    """Agent 工厂 - 根据配置自动创建 Agent 实例"""

    # Agent 类注册表
    _agent_classes = {}

    @classmethod
    def register(cls, agent_class_name: str):
        """装饰器：注册 Agent 类"""
        def decorator(agent_class):
            cls._agent_classes[agent_class_name] = agent_class
            logger.debug(f"Agent 类已注册: {agent_class_name}")
            return agent_class
        return decorator

    @classmethod
    def create_agent(cls, agent_id: str, agent_config: Dict[str, Any],
                     api_key: str, base_url: str, napcat_client):
        """
        创建 Agent 实例

        Args:
            agent_id: Agent ID
            agent_config: Agent 配置
            api_key: API Key
            base_url: API Base URL
            napcat_client: NapCat 客户端实例

        Returns:
            Agent 实例

        Raises:
            ValueError: 如果 Agent 类型未注册
        """
        agent_class_name = agent_config.get("class", "")

        if not agent_class_name:
            raise ValueError(f"Agent 配置缺少 'class' 字段: {agent_id}")

        if agent_class_name not in cls._agent_classes:
            raise ValueError(f"未知的 Agent 类型: {agent_class_name}")

        agent_class = cls._agent_classes[agent_class_name]

        # 合并配置
        config = {**agent_config, **agent_config.get("config", {})}

        # 创建实例
        agent = agent_class(
            agent_id=agent_id,
            config=config,
            api_key=api_key,
            base_url=base_url,
            napcat_client=napcat_client
        )

        logger.debug(f"Agent 实例已创建: {agent_id} ({agent.agent_name})")
        return agent

    @classmethod
    def load_all_agents(cls, agents_config: Dict[str, Any], api_key: str,
                       base_url: str, napcat_client, agent_manager):
        """
        加载所有 Agent

        Args:
            agents_config: Agent 配置字典
            api_key: API Key
            base_url: API Base URL
            napcat_client: NapCat 客户端实例
            agent_manager: Agent 管理器实例

        Returns:
            成功加载的 Agent 数量
        """
        loaded_count = 0

        for agent_id, agent_config in agents_config.items():
            # 跳过已禁用的 Agent
            if not agent_config.get("enabled", True):
                logger.info(f"跳过已禁用的 Agent: {agent_id}")
                continue

            try:
                # 创建 Agent 实例
                agent = cls.create_agent(
                    agent_id=agent_id,
                    agent_config=agent_config,
                    api_key=api_key,
                    base_url=base_url,
                    napcat_client=napcat_client
                )

                # 注册到 AgentManager
                agent_manager.register_agent(agent_id, agent)
                logger.success(f"Agent 已加载: {agent_id} ({agent.agent_name})")
                loaded_count += 1

            except Exception as e:
                logger.error(f"加载 Agent 失败 ({agent_id}): {e}", exc_info=True)

        return loaded_count
