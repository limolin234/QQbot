# Forward Agent

## 工作流（简版）
- 触发入口：群消息事件
- 调度流程：
  - `scheduler.enqueue_forward_by_monitor_group` 检查监控群号
  - 命中后投递 `TaskType.FROWARD`
  - `handler.handle_task` 调用 `run_forward_graph`
  - `should_forward=true` 时，私聊转发给主人 QQ

## 配置项（workflows/agent_config.yaml）
- `forward_config.config.model`：转发判定模型 ID
- `forward_config.config.temperature`：采样温度
- `forward_config.config.monitor_group_qq_number`：监控群号列表
- `forward_config.config.forward_decision_prompt`：转发判定提示词

