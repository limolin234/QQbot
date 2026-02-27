# Dida 双模型重构需求文档（PRD）

## 1. 背景与动机

当前 `dida_agent` 使用同一次大模型调用同时生成：

- 面向用户的回复文本（带“十一”口癖/动作/语气）
- 面向系统的结构化动作 `DidaAction`（create/list/update/complete/delete）

并在执行 Dida API 后，将执行结果以字符串拼接方式附加到原回复文本后发送（AI 不参与基于执行结果的二次组织）。

这会导致：

- 任务管理提示词与角色扮演提示词混杂，模型更容易偏离“必须输出 task_id”的要求
- 同名任务/模糊指令时，模型可能只输出 `title`，后端会回退到标题匹配，出现歧义或失败
- 回复无法真正“根据执行成功/失败原因”进行自然语言的二次表达（只是拼接）

本 PRD 目标：将“任务理解/动作生成”和“对话回复生成”解耦为两次模型调用，并将任务上下文以结构化 JSON 注入第一阶段，提升可控性与可靠性。

## 2. 现状梳理（基于当前代码）

### 2.1 关键文件

- `d:\Microsoft VS Code\PYTHON\QQbot\workflows\dida\dida_agent.py`
- `d:\Microsoft VS Code\PYTHON\QQbot\workflows\dida\dida_scheduler.py`
- `d:\Microsoft VS Code\PYTHON\QQbot\workflows\dida\dida_service.py`
- `d:\Microsoft VS Code\PYTHON\QQbot\workflows\agent_config.yaml`
- `d:\Microsoft VS Code\PYTHON\QQbot\data\dida_context.json`（scheduler 定时写入的任务上下文缓存）

### 2.2 当前调用链（简化）

1. 消息进入 dida_agent，经过规则过滤与冷却聚合  
   入口：`enqueue_dida_agent_if_monitored()` → `DidaAgentDispatcher.enqueue_if_monitored()`  
   文件：[dida_agent.py](file:///d:/Microsoft%20VS%20Code/PYTHON/QQbot/workflows/dida/dida_agent.py)

2. 若命中规则、需要回复：调用一次大模型生成 `reply_text` + `dida_action`  
   入口：`run_dida_agent_pipeline()` → `DidaAgentDecisionEngine.generate_reply_text()`  
   文件：[dida_agent.py](file:///d:/Microsoft%20VS%20Code/PYTHON/QQbot/workflows/dida/dida_agent.py#L1035-L1347)

3. 若存在 `dida_action`：调用 scheduler 执行 Dida API，得到 dida_response 字符串  
   入口：`dida_scheduler.execute_action()`  
   文件：[dida_scheduler.py](file:///d:/Microsoft%20VS%20Code/PYTHON/QQbot/workflows/dida/dida_scheduler.py#L247-L536)

4. 最终发送：`reply_text` 与 `dida_response` 以 `\n` 拼接后发送（不会二次调用 AI）  
   文件：[dida_agent.py](file:///d:/Microsoft%20VS%20Code/PYTHON/QQbot/workflows/dida/dida_agent.py#L430-L585)

### 2.3 现状的任务定位策略

对于 `complete/delete/update`：

- 若模型输出 `task_id`：后端优先按 `task_id` 精确匹配（唯一）  
- 若模型未输出 `task_id` 但输出 `title`：后端按 `title` 纯相等匹配，并返回第一个命中的任务

实现：`DidaScheduler._find_task_obj()`  
文件：[dida_scheduler.py](file:///d:/Microsoft%20VS%20Code/PYTHON/QQbot/workflows/dida/dida_scheduler.py#L181-L226)

## 3. 需求目标

### 3.1 核心目标

1. 提示词拆分：将“任务管理（动作生成）提示词”与“回复风格（十一设定）提示词”拆开
2. 双模型两阶段：
   - 阶段 A：任务理解模型（Model A）只负责生成 `DidaAction`（优先/强制输出 `task_id` 等）
   - 阶段 B：回复生成模型（Model B）基于“执行结果 + 语言风格提示词”生成最终回复
3. 结构化上下文注入：阶段 A 注入 JSON 任务上下文 + 当前指令，不再用纯文本列表
4. 动作执行可控性：对 update/complete/delete 等“操作型动作”，要求不允许仅靠标题执行

### 3.2 非目标（本阶段不做）

- 不改 Dida OAuth/Token 存储逻辑
- 不引入新的外部数据库/缓存（仍使用 `data/dida_context.json` 作为上下文来源）
- 不调整 ncatbot 的消息接入、群控逻辑

## 4. 术语与角色

- Model A（Action LLM）：根据自然语言 + 结构化任务上下文生成 `DidaAction`
- Model B（Reply LLM）：根据执行结果 + 风格提示词生成最终 `reply_text`
- DidaAction：结构化动作对象（现有 Pydantic 模型）
- Task Context：`dida_context.json` 中缓存的任务集合（含 id/title/projectId/due 等）

## 5. 新方案：总体流程设计

### 5.1 新时序（推荐）

1. dida_agent 判定需要处理（沿用现有 `trigger_mode` / `ai_decision_prompt`）
2. dida_agent 读取 task context（从 `data/dida_context.json`，只取当前用户/管理员指定用户）
3. 调用 Model A（Action）：
   - 输入：结构化 JSON（含 user 指令、ts、chat 元信息、任务列表 JSON）
   - 输出：`DidaAction` 或 `null`（无法解析/需要澄清）
4. dida_agent 调用 `dida_scheduler.execute_action(action=...)` 执行 Dida API
5. 调用 Model B（Reply）：
   - 输入：用户原始指令 + Model A 的动作（可隐藏敏感字段）+ Dida 执行状态（成功/失败/原因）
   - 输出：最终发送的 `reply_text`（完全按“十一风格”）
6. 发送消息

### 5.2 关键约束（保证“只用 id 执行”）

对于 `action_type in {"complete","delete","update"}`：

- 若 Model A 未提供 `task_id`：不得执行 Dida API  
  必须返回一种“澄清”型结果（由 Model B 输出给用户），例如：  
  “主人我找到两个‘做大创’，你要完成哪一个呀汪~（02-25 / 02-26）”

对于 `list/create`：

- list：允许不填 project_id（后端可列全项目）
- create：需要 title；due_date 规则沿用当前配置

## 6. 提示词与配置拆分（agent_config.yaml）

### 6.1 当前问题

`dida_agent_config.rules[*].reply_prompt` 同时包含：

- 大量角色设定/口癖/动作
- 任务管理策略（生成 dida_action、强调 task_id）

这会导致 Model 输出在“风格优先”时削弱“结构化动作优先”的确定性。

### 6.2 目标配置形态（建议）

在 `dida_agent_config.rules[*]` 下拆分为三段：

1. `ai_decision_prompt`（保持不变，用于 should_reply 判定）
2. `action_prompt`（新增，仅用于 Model A）
3. `reply_prompt`（保留，但只用于 Model B 的风格/措辞，不再要求输出 dida_action）

建议新增字段：

- `action_model`：Model A 使用的模型名（可与 `model` 不同）
- `reply_model`：Model B 使用的模型名（可与 `model` 不同）
- `action_temperature` / `reply_temperature`：分别控制温度
- `action_enabled`：是否启用 Model A（便于灰度）

兼容策略：

- 若 `action_prompt` 缺失，则回退为旧单模型模式（保持线上可用）

## 7. 数据契约（强约束输入输出）

### 7.1 Model A 输入：ActionRequest（结构化）

建议形态（示例）：

```json
{
  "meta": {
    "chat_type": "group",
    "group_id": "499616852",
    "user_id": "1830740938",
    "user_name": "沂水何谙",
    "ts": "2026-02-26T18:55:36+08:00"
  },
  "instruction": {
    "raw_message": "...原消息...",
    "cleaned_message": "完成做大创"
  },
  "tasks": [
    {
      "id": "67bf....",
      "title": "做大创",
      "projectId": "xxxx",
      "project": "📖学习安排",
      "due": "2026-02-26T11:00:00.000Z",
      "isAllDay": false,
      "status": "0"
    }
  ]
}
```

注意：

- `tasks` 必须来自 `dida_context.json`，并尽可能包含 `projectId`
- Model A 必须在“任务唯一定位”时选择 `id`

### 7.2 Model A 输出：DidaAction（严格）

- 对操作型动作：必须输出 `task_id`，建议同时输出 `project_id`
- 对歧义/无法定位：输出 `dida_action=null`，并通过字段说明需要澄清

建议扩展新的结构化输出模型（不破坏现有 `DidaAction`）：

```json
{
  "dida_action": {
    "action_type": "complete",
    "task_id": "67bf....",
    "project_id": "xxxx"
  },
  "need_clarification": false,
  "clarification_question": ""
}
```

### 7.3 Dida 执行结果：ActionExecutionResult

当前 `dida_scheduler.execute_action` 返回字符串，不利于 Model B 结构化理解。建议新增内部使用的结构化结果，至少包含：

- `ok: bool`
- `action_type`
- `task_title`（可选）
- `task_id`（内部字段，给 Model B 时应脱敏或不提供）
- `message_for_user`（可选：可复用原字符串）
- `error_code` / `error_message`（失败时）

## 8. 回复生成（Model B）设计

### 8.1 Model B 输入：ReplyRequest

```json
{
  "meta": { "...同上..." },
  "instruction": { "cleaned_message": "完成做大创" },
  "action": {
    "action_type": "complete",
    "resolved": true,
    "task_title": "做大创",
    "task_due_display": "02-26 19:00"
  },
  "execution": {
    "ok": true,
    "message_for_user": "✅ 已完成任务：做大创"
  }
}
```

说明：

- 不向 Model B 暴露 `task_id/project_id`（避免泄露到回复）
- 仅给可展示信息（title、due 展示、成功/失败、简短原因）

### 8.2 Model B 输出：ReplyResponse

仅包含最终要发送的 `reply_text`（风格由拆分后的 `reply_prompt` 控制）。

## 9. 交互策略（同名/歧义/缺信息）

### 9.1 同名任务

当用户说“完成做大创”，但任务上下文中存在多个同名：

- Model A 必须返回 `need_clarification=true`
- `clarification_question` 应给出可选项（但不泄露 ID），例如按 due / project 展示
- 不执行 Dida API
- Model B 根据 `clarification_question` 输出自然语言追问

### 9.2 上下文缺失（dida_context 为空/过期）

当上下文为空或不包含目标任务：

- Model A 输出 `need_clarification=true`，建议引导用户先“查看任务列表”或触发一次 list
- 可选：由系统自动先执行 list（但需注意成本与响应延迟，本 PRD 默认先不自动）

### 9.3 权限与代办（管理员 @ 他人）

现有逻辑支持 `target_user_id`（管理员代办/查看他人任务），需保持：

- Model A 输入中应包含“被 @ 的用户列表”
- Model A 输出若涉及他人任务，必须填 `target_user_id`
- 任务上下文的选择范围：管理员可取被 @ 用户的 tasks；普通用户仅取自己

## 10. 日志与可观测性（需求）

为便于排障与迭代，需要新增/补齐事件埋点：

- Model A 调用开始/结束：model、耗时、输出是否包含 task_id、need_clarification
- Dida 执行结果：ok/失败原因/错误码
- Model B 调用开始/结束：model、耗时、reply_length

注意日志中不得输出 access_token/refresh_token 等敏感信息。

## 11. 兼容与迁移计划

### 11.1 强制双模型与配置校验

- 系统始终启用双模型模式，不提供单模型回退路径
- 若任一 rule 缺少 `action_prompt`，应在启动/加载配置阶段直接报错并拒绝运行该 rule
- 报错信息需包含可定位的上下文（rule 的 chat_type + number），例如：`dida_agent_config.rules[x] missing action_prompt (chat_type=group number=499616852)`

### 11.2 灰度建议

- 先在一个群号 rule 上开启 `action_enabled: true`
- 观察日志中 `need_clarification` 比例、`task_id` 命中率、误操作率

## 12. 验收标准（Acceptance Criteria）

1. 对 complete/delete/update：系统在任何情况下都不会仅凭 title 执行
2. 当同名任务存在时：必然触发澄清流程（不执行），回复可读且符合“十一风格”
3. 当能唯一定位时：Model A 必须输出 task_id，scheduler 执行成功，Model B 回复中不泄露 ID
4. list/create 正常可用，与现有功能一致或更好
5. 发生 Dida API 失败时：Model B 能生成基于失败原因的自然语言回复（不是简单拼接）

## 13. 测试与验证（建议）

- 单元测试（建议新增）：
  - Model A 输出缺 task_id 时，complete/delete/update 不执行
  - 同名任务时，need_clarification 流程返回预期结构
- 集成测试（手工/回归）：
  - “完成做大创”存在两条同名任务 → 追问选择
  - “完成 02-26 的做大创” → 定位正确任务并完成
  - “查看@某人 的任务”管理员权限生效

## 14. 开发拆分（实现建议）

推荐按模块落地：

1. 在 `dida_agent.py` 中新增 Action LLM 调用函数：只产出动作与澄清信息
2. 在 `dida_scheduler.py` 中新增结构化执行结果返回（内部），同时保留原字符串以兼容
3. 在 `dida_agent.py` 中新增 Reply LLM 调用函数：输入执行结果与风格 prompt 生成回复
4. 扩展 `agent_config.yaml`：新增 `action_prompt/action_model/...` 等字段，并实现 fallback
