# Dida 双模型重构执行计划（Task List / File Structure）

本计划基于：[dida_improve.md](file:///d:/Microsoft%20VS%20Code/PYTHON/QQbot/dida_improve.md)。

## 1. 范围检查：需要改哪些地方

### 1.1 代码（workflows/dida）

- `dida_agent.py`
  - 需要从“单次模型调用（reply_text + dida_action）”改为“两阶段模型调用（Action → Execute → Reply）”
  - 需要新增配置校验：缺少 `action_prompt` 直接报错（PRD 11.1）
  - 需要将任务上下文注入从“纯文本列表”改为“结构化 JSON tasks”
  - 需要新增 Model A / Model B 的结构化输出模型（Pydantic）
  - 需要新增“操作型动作必须有 task_id”的强约束（complete/delete/update）

- `dida_scheduler.py`
  - 需要补齐/统一 `data/dida_context.json` 的字段 schema
    - `poll_once()` 写入的 tasks 已包含 `projectId/isAllDay`
    - `list` 分支写入的 tasks 当前缺少 `projectId/isAllDay`，需对齐，否则 Model A 输入不稳定
  - 需要提供可供 Model B 消费的“结构化执行结果”
    - 当前 `execute_action()` 返回 string，不利于 Model B 做二次组织
    - 建议新增 `execute_action_structured()`（不改旧函数签名）或在 `execute_action()` 内部生成结构化结果并返回（会影响调用方）

- `dida_service.py`
  - 基本无需改动（仅负责 OpenAPI 请求）

### 1.2 配置（agent_config.yaml.example）

- `dida_agent_config.rules[*]` 需要新增并强制要求：
  - `action_prompt`：Model A 的系统提示词（只负责生成动作/澄清）
  - `reply_prompt`：Model B 的系统提示词（只负责生成最终回复文本）
  - 可选：`action_model/reply_model`、`action_temperature/reply_temperature`
- `reply_prompt` 不再承担“输出 dida_action”的职责（避免与风格混杂）

## 2. 架构/接口影响评估（是否破坏现有功能）

### 2.1 运行期行为变化（breaking behavior）

- 对 `complete/delete/update`：
  - 旧行为：允许仅凭 `title` 执行（scheduler 回退标题匹配）
  - 新行为：强制 `task_id`，否则不执行并走澄清回复
  - 结论：这会改变一部分“以前能执行但不安全”的指令表现，但符合新需求与验收标准

- 对配置：
  - 旧配置只有 `reply_prompt` 也能跑
  - 新行为：缺 `action_prompt` 直接报错并拒绝运行对应 rule
  - 结论：这是配置层面的 breaking change；需要同步更新 `workflows/agent_config.yaml`

### 2.2 代码接口变化（尽量不破坏现有架构）

建议策略：优先“新增接口”，避免修改已被调用的函数签名。

- `DidaScheduler.execute_action(...) -> str`
  - 现状：仅被 `dida_agent.py` 调用（当前代码库内）
  - 建议：保留不变，新增：
    - `execute_action_structured(...) -> dict[str, Any]`（或 Pydantic/TypedDict）
  - 好处：不破坏现有调用点；便于逐步迁移与回滚

- `DidaAgentDecisionEngine.generate_reply_text(...)`
  - 将被拆分为：
    - `generate_action(...)`（Model A）
    - `generate_reply(...)`（Model B）
  - 可保留原方法但不再使用；或改名并保留兼容入口（内部调用新流程）
  - 风险：文件内调用点较集中，影响面可控

## 3. 文件结构（改动清单）

### 3.1 必改文件

- `d:\Microsoft VS Code\PYTHON\QQbot\workflows\dida\dida_agent.py`
- `d:\Microsoft VS Code\PYTHON\QQbot\workflows\dida\dida_scheduler.py`
- `d:\Microsoft VS Code\PYTHON\QQbot\workflows\agent_config.yaml.example`
- `d:\Microsoft VS Code\PYTHON\QQbot\workflows\agent_config.yaml`（真实运行配置，需同步升级，否则会触发新校验报错）

### 3.2 可选新增文件（用于解耦，非必须）

若 `dida_agent.py` 体积继续增长，建议拆出：

- `workflows/dida/dida_action_llm.py`：Model A（Action）封装
- `workflows/dida/dida_reply_llm.py`：Model B（Reply）封装
- `workflows/dida/dida_types.py`：Pydantic/TypedDict 数据契约

（也可先全部写在 `dida_agent.py`，后续再拆分）

## 4. Coding Tasks（Tickets）与执行顺序

下面的 Ticket 按依赖顺序排列（先配置与数据契约 → 再引擎/执行 → 再回归与测试）。

### Ticket 01：更新示例配置 schema（强制双模型字段）

- 目标：更新 `agent_config.yaml.example` 中 `dida_agent_config`，新增 `action_prompt` 等字段，并将原 `reply_prompt` 的“任务管理输出要求”迁移到 `action_prompt`
- 主要改动文件：
  - `workflows/agent_config.yaml.example`
- 接口影响：
  - 无代码接口变更，但明确未来运行配置必须包含 `action_prompt`

### Ticket 02：实现 dida_agent 配置校验（缺 action_prompt 抛错）

- 目标：在 dida_agent 加载/启动时检查：
  - `dida_enabled=true` 的 rule 必须存在非空 `action_prompt` 与 `reply_prompt`
  - 报错信息包含 `chat_type + number`
- 主要改动文件：
  - `workflows/dida/dida_agent.py`
- 风险：
  - 现网 `workflows/agent_config.yaml` 未更新会直接报错（预期行为）

### Ticket 03：统一 dida_context.json 的任务 schema

- 目标：确保 `data/dida_context.json` 中 tasks 字段稳定包含：
  - `id/title/project/projectId/due/isAllDay`（至少这些）
- 主要改动文件：
  - `workflows/dida/dida_scheduler.py`
- 依赖：
  - 无
- 备注：
  - 需要同步修复 `list` 分支写入 context 时缺字段的问题

### Ticket 04：定义 Model A / Model B 数据契约（Pydantic 模型）

- 目标：在代码层定义并复用：
  - `ActionRequest`（结构化输入：meta + instruction + tasks）
  - `ActionLLMResult`（含 dida_action + need_clarification + clarification_question）
  - `ReplyRequest`（输入：指令 + action 摘要 + execution 结果）
  - `ReplyResponse`（输出：reply_text）
- 主要改动文件：
  - `workflows/dida/dida_agent.py`（或可选新文件 `dida_types.py`）
- 风险：
  - 需要确保结构化输出与所用模型兼容（已有 `with_structured_output` 经验）

### Ticket 05：实现 Model A（Action LLM）调用与结构化 JSON 注入

- 目标：
  - 从 `data/dida_context.json` 按用户/管理员范围提取 tasks
  - 构造 `ActionRequest` JSON 作为 HumanMessage
  - 使用 `action_prompt` 作为 SystemMessage
  - 产出 `ActionLLMResult`
- 主要改动文件：
  - `workflows/dida/dida_agent.py`
- 依赖：
  - Ticket 02、03、04

### Ticket 06：强制 task_id 执行策略（防止标题回退）

- 目标：
  - 在 dida_agent 中：若 action_type 属于 complete/delete/update 且缺 task_id：
    - 不调用 scheduler
    - 生成“需要澄清”的 execution 状态交给 Model B
  - 可选：在 dida_scheduler 中也加一道防线（即使收到 title 也拒绝执行）
- 主要改动文件：
  - `workflows/dida/dida_agent.py`
  - （可选）`workflows/dida/dida_scheduler.py`
- 依赖：
  - Ticket 05
- 接口影响：
  - 行为层 breaking（符合需求）

### Ticket 07：提供结构化执行结果（给 Model B 使用）

- 目标：
  - 为 Dida API 执行结果提供结构化对象：
    - `ok/action_type/message_for_user/error_message/...`
  - 推荐实现方式：
    - 新增 `DidaScheduler.execute_action_structured(...)`，内部复用现有逻辑
    - 旧 `execute_action(...) -> str` 保留，用于兼容/工具命令输出
- 主要改动文件：
  - `workflows/dida/dida_scheduler.py`
  - `workflows/dida/dida_agent.py`（调用新接口）
- 依赖：
  - Ticket 04、06
- 接口影响：
  - 新增接口，不破坏旧接口

### Ticket 08：实现 Model B（Reply LLM）基于执行结果生成最终回复

- 目标：
  - 构造 `ReplyRequest`（不包含 task_id/project_id）
  - 使用拆分后的 `reply_prompt`（仅风格/措辞）
  - 输出最终 `reply_text`
- 主要改动文件：
  - `workflows/dida/dida_agent.py`
- 依赖：
  - Ticket 07

### Ticket 09：串联主流程（Action → Execute → Reply）

- 目标：替换 `run_dida_agent_pipeline` 内的生成逻辑：
  - should_reply 判定仍按原逻辑
  - 若 dida_enabled：
    - 调 Model A 得到 action 或澄清
    - 执行/构造 execution
    - 调 Model B 生成最终回复
- 主要改动文件：
  - `workflows/dida/dida_agent.py`
- 依赖：
  - Ticket 05、06、07、08

### Ticket 10：可观测性与日志补齐

- 目标：新增埋点字段：
  - Model A/model B 耗时、输出是否含 task_id、need_clarification、执行 ok/失败原因
- 主要改动文件：
  - `workflows/dida/dida_agent.py`
  - （可选）`workflows/dida/dida_scheduler.py`
- 依赖：
  - Ticket 09

### Ticket 11：回归与测试用例（最小集合）

- 目标：覆盖验收标准的关键路径：
  - 同名任务 → 触发澄清（不执行）
  - 唯一定位 → 输出 task_id → 执行成功 → 回复不泄露 ID
  - 执行失败 → 回复基于失败原因自然表达
- 主要改动文件：
  - 视项目测试框架而定（若已有 pytest 则新增 pytest；否则用 unittest 并提供可运行入口）
- 依赖：
  - Ticket 09

## 5. 推荐落地顺序（总览）

1. Ticket 01（示例配置） → Ticket 02（强校验）
2. Ticket 03（context schema） → Ticket 04（数据契约）
3. Ticket 05（Model A） → Ticket 06（强制 task_id）
4. Ticket 07（结构化执行结果） → Ticket 08（Model B）
5. Ticket 09（主流程串联） → Ticket 10（日志） → Ticket 11（测试回归）
