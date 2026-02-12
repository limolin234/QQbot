# Auto Reply Agent

## 工作流（简版）
- 触发入口：
  - 群聊：`main.on_group_message` -> `enqueue_auto_reply_if_monitored(..., "group")`
  - 私聊：`main.on_private_message` -> `enqueue_auto_reply_if_monitored(..., "private")`
- 调度流程：
  - 命中监控目标后由 `workflows.auto_reply` 直接提交执行（`submit_agent_job(...)`）
  - 执行 `run_auto_reply_pipeline`
  - 管道步骤：
    1) 加载最近上下文（来自 `message.jsonl`）
    2) 判定是否回复（表达式 + 可选 AI 判定）
    3) 生成回复文本（LangGraph）
    4) 发送消息（群聊/私聊）
  - 全链路写入 `logs/agent_events.jsonl`

## 配置项（workflows/agent_config.yaml）
- `auto_reply_config.config.model`：判定/生成共用模型 ID
- `auto_reply_config.config.temperature`：采样温度
- `auto_reply_config.config.rules`：规则列表（每条规则对应一个目标会话）

### rules 元素字段
- `enabled`：是否启用规则
- `chat_type`：`group` 或 `private`
- `number`：群号或私聊用户号
- `trigger_mode`：触发表达式（支持 `&&` / `||` / `!` / `()`）
  - 原子条件：`at_bot` / `keyword` / `always` / `ai_decide`
- `keywords`：关键词列表（供 `keyword` 使用）
- `ai_decision_prompt`：AI 判定提示词（供 `ai_decide` 使用）
- `reply_prompt`：AI 生成回复提示词

## 可选配置（代码支持，未写也可用默认）
- `auto_reply_config.config.context_history_limit`：上下文最大条数（默认 `12`）
- `auto_reply_config.config.context_max_chars`：上下文最大字符数（默认 `1800`）
- `auto_reply_config.config.context_window_seconds`：只读取最近 N 秒上下文（默认 `1800`）
- `auto_reply_config.config.min_reply_interval_seconds`：同会话最小回复间隔（默认 `300`）
- `auto_reply_config.config.flush_check_interval_seconds`：pending 扫描周期（默认 `10`）
- `auto_reply_config.config.pending_expire_seconds`：pending 过期时长（默认 `3600`）
- `auto_reply_config.config.pending_max_messages`：冷却期内 pending 最大累计条数（默认 `50`）
- `auto_reply_config.config.bypass_cooldown_when_at_bot`：群聊 @bot 时是否绕过冷却（默认 `true`）

## 限流调试日志
- 写入 `logs/agent_events.jsonl`（`agent=auto_reply`, `task_type=AUTO_REPLY`）。
- `stage=throttle_enqueue_now`：本条消息立即进入 `AUTO_REPLY` 队列。
- `stage=throttle_queued_pending`：命中冷却，本条进入 pending 累计。
- `stage=throttle_pending_flush`：冷却到期，pending 被统一投递。
- `stage=throttle_pending_expired`：pending 超过保留时长被丢弃。
- `stage=throttle_bypass_cooldown`：命中 @bot，跳过冷却立即处理。
