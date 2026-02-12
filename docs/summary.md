# Summary Agent

## 工作流（简版）
- 触发入口：
  - 私聊指令 `/summary`（手动）
  - 每日定时任务（自动）
- 调度流程：
  - `workflows.summary.daily_summary` 读取 `message.jsonl`
  - 分组/分块后直接通过 `submit_agent_job(...)` 执行摘要图
  - 结果格式化后私聊发送给主人 QQ

## 配置项（workflows/agent_config.yaml）
- `summary_config.config.model`：摘要模型 ID
- `summary_config.config.temperature`：采样温度
- `summary_config.config.max_line_chars`：单行最大长度
- `summary_config.config.max_lines`：单批最多处理行数
- `summary_config.config.summary_chat_scope`：消息范围
  - `group` / `private` / `all`
