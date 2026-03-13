# 统一定时任务调度模块 PRD

## 背景与目标
当前 QQBot 项目中，定时任务逻辑分散在不同模块中：
- `daily_summary` 使用 `aiocron` 硬编码在 `workflows/summary.py` 中。
- `dida_scheduler` 使用 `while True` + `sleep` 硬编码在 `workflows/dida/dida_scheduler.py` 中。

这种分散导致：
1.  **维护困难**：查看所有定时任务需要遍历代码。
2.  **扩展性差**：新增简单定时任务（如“每天早上问好”）需要编写 Python 代码。
3.  **配置不透明**：定时规则（如 `0 22 * * *`）埋在代码里，非开发者难以修改。

**目标**：构建一个统一的 **Scheduler Agent**（或模块），通过单一 YAML 配置文件管理所有定时任务，支持 Cron 表达式和定时间隔（Interval），并能灵活调用已注册的系统功能。

## User Stories

### P0 优先级 (Core)
1.  **统一配置**：作为管理员，我希望在 `agent_config.yaml`（或独立 `scheduler.yaml`）中定义所有定时任务，无需修改 Python 代码。
2.  **Cron 支持**：我希望支持标准的 Cron 表达式（如 `0 22 * * *`）来执行每日/每周任务。
3.  **多步执行 (Steps)**：我希望一个定时任务可以顺序执行多个动作（例如：先发“早安”，再发“任务列表”）。
4.  **现有功能迁移**：我希望现有的“每日总结”和“滴答清单轮询”能迁移到该框架下，行为保持不变。

### P1 优先级 (Enhanced)
1.  **参数化任务报告**：我希望通过配置 Action 参数（如 `project_id`），灵活控制发送哪些任务列表。
2.  **开关控制**：我希望通过 `enabled: false` 快速暂停某个任务。

## 详细功能需求 (Functional Requirements)

### 1. 配置文件定义
在 `workflows/agent_config.yaml` 中新增 `schedules` 顶级字段（或独立文件），结构升级为支持 `steps`：

```yaml
schedules:
  - name: "早安与任务提醒"
    type: "cron"
    expression: "0 8 * * *"
    enabled: true
    steps:
      - action: "core.send_group_msg"
        params:
          group_id: "12345678"
          message: "🌞 早安！新的一天开始了，请查看今日任务："
      
      - action: "dida.push_task_list"
        params:
          group_id: "12345678"
          user_qq: "123456"     # 过滤特定用户的任务
          day_range: "today"    # today, tomorrow, overdue, all
          limit: 10

  - name: "每日总结"
    type: "cron"
    expression: "0 22 * * *"
    steps:
      - action: "summary.daily_report"
        params:
          mode: "auto"

  - name: "滴答清单轮询"
    type: "interval"
    seconds: 60
    steps:
      - action: "dida.poll"
        params:
          project_ids: ["all"]
```

### 2. 调度管理器 (Scheduler Manager)
- **职责**：
    - 启动时读取配置。
    - 维护任务注册表（Action Registry）。
    - 初始化 `aiocron`（针对 cron）和 `asyncio` 循环（针对 interval）。
    - **Step 执行逻辑**：按顺序执行 `steps` 列表中的 Action。如果前一个 Step 报错，默认记录日志并继续执行下一个（除非配置 `fail_fast: true`）。

### 3. 动作注册表 (Action Registry)
扩展标准 Action 库，以支持模块化组合：

| Action ID | 对应功能 | 参数说明 |
| :--- | :--- | :--- |
| `core.send_group_msg` | 发送群消息 | `group_id`, `message` |
| `core.send_private_msg` | 发送私聊 | `user_id`, `message` |
| `dida.push_task_list` | **[新增]** 推送任务列表 | `target_id` (群/人), `user_qq` (筛选归属), `day_range` (时间范围) |
| `summary.daily_report` | 执行每日总结 | `mode` |
| `dida.poll` | 执行滴答轮询 | `project_ids` |

## 验收标准 (Acceptance Criteria)

### Scenario 1: 组合任务执行
- **Given** 配置了一个 Cron 任务，包含 Step 1 (发送问候) 和 Step 2 (发送任务列表)。
- **When** 触发时间到达。
- **Then** Bot 应先发送问候语，紧接着发送该用户的今日任务列表。

### Scenario 2: 动态参数注入
- **Given** 配置 `dida.push_task_list` 参数为 `day_range: "overdue"`.
- **When** 任务执行时。
- **Then** 输出的内容应仅包含已过期的任务，不包含未来的任务。

### Scenario 3: 容错性
- **Given** 组合任务中 Step 1 失败（如网络波动）。
- **When** 任务执行时。
- **Then** 系统记录 Step 1 错误，但 Step 2 仍尝试执行（保证重要通知不被非关键错误阻塞）。

## 数据字段定义

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `name` | string | 是 | 任务唯一标识/描述 |
| `type` | enum | 是 | `cron` 或 `interval` |
| `expression` | string | 否 | Cron 表达式（`type=cron` 时必填） |
| `seconds` | int | 否 | 间隔秒数（`type=interval` 时必填） |
| `steps` | list | 是 | 执行步骤列表 |
| `steps[].action` | string | 是 | 注册的动作标识符 |
| `steps[].params` | dict | 否 | 传递给动作函数的参数 |
| `enabled` | bool | 否 | 默认为 true |

---

## 当前进度 (Current Status) - 2026-03-13

### ✅ 已完成 (Completed)
1.  **核心架构**: `SchedulerManager` 和 `ActionRegistry` 已实现，支持 Cron (aiocron) 和 Interval (asyncio) 任务。
2.  **基础 Action**: `core.send_group_msg` 和 `core.send_private_msg` 已实现并注册。
3.  **系统集成**: `main.py` 已集成 Scheduler 启动流程。
4.  **业务适配 (进行中)**: `summary.py` 和 `dida_scheduler.py` 已添加 Action 注册代码。

### 🚧 阻塞点与待办 (Blockers & Todo)
1.  **配置加载器缺失**: `agent_config_loader.py` 中缺少 `is_scheduler_enabled()` 函数，导致业务模块导入报错。
2.  **Dida 任务推送未实现**: `dida.push_task_list` Action 尚未实现，无法配置早安任务。
3.  **配置文件未更新**: `agent_config.yaml` 尚未添加 `schedules` 字段。

---

## 下一步实施计划 (Updated)
1.  [x] 创建 `workflows/scheduler/` 目录。
2.  [x] 实现 `scheduler_manager.py` (现为 `manager.py`)。
3.  [x] 定义 `ActionRegistry` 并注册基础功能。
4.  [x] 修改 `main.py` 引入 Scheduler 启动逻辑。
5.  [ ] **修复基础设施**: 在 `workflows/agent_config_loader.py` 中实现 `is_scheduler_enabled()`。
6.  [ ] **实现核心业务 Action**:
    -   实现 `dida.push_task_list` (支持按 `day_range` 筛选任务)。
    -   完善 `dida.poll` 的参数化调用。
7.  [ ] **配置迁移**: 在 `agent_config.yaml` 中添加 `schedules` 字段，并配置“早安”、“每日总结”和“滴答轮询”任务。
8.  [ ] **验证与清理**: 启动 Bot 验证调度器工作正常，移除旧的硬编码调度逻辑。

请确认是否按照此 PRD 进行开发？你可以回复“**开始执行**”或提出修改意见。