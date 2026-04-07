# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 全局规则

## 语言规则

**所有回复必须使用中文作为主要语言。**

无论 skill 文件、项目规则或其他配置文件使用何种语言（英文或其他），OpenCode 的回复都必须以中文为主。

### 适用范围

- 所有对话回复
- 任务说明和计划
- 代码注释（除非项目明确要求英文注释）
- 错误解释和调试信息
- 文档和说明

### 例外情况

- 代码本身（变量名、函数名等保持原样）
- 技术术语可保留英文（如 API、HTTP、JSON 等）
- 用户明确要求使用其他语言时

## Working Agreements

- 对于编辑请求，可直接执行文件修改；无需在修改前单独贴出 Diff（用户会在编辑请求流程中审核 Diff）。
- 在执行非只读的终端命令前，先解释该命令的作用。
- 如果任务涉及多个文件，先提供一个修改计划。
- 必须使用 Codex 原生编辑工具进行代码修改（使用 `apply_patch`）；禁止使用 `cat > 文件`、heredoc 覆盖写入等方式修改代码文件。
- 所有代码文件写入必须走 `apply_patch`（包括新增/修改/删除）；禁止使用 `cat`/`tee`/重定向/`sed -i`/heredoc 等直接写文件方式。
- 必须通过平台提供的工具通道调用 `apply_patch`；禁止将 `apply_patch` 作为普通 shell 命令经 `exec_command` 间接调用。
- 避免过度解耦：不要为了"形式上的模块化"拆出大量 `_` 开头私有小函数；仅当逻辑被复用、明显增大可读性、或便于测试时再抽取函数。单次使用且逻辑很短的处理应优先保留在主流程中。

---

## 项目概述

基于 Python 的 QQ 聊天机器人，采用模块化 Agent 架构，利用 LangGraph 构建工作流，支持自动回复、消息转发、每日摘要、滴答清单任务管理。

## 常用命令

```bash
# 启动机器人
python main.py

# 启动 Config Studio 可视化配置面板（前端 + 后端）
./start_studio.sh

# 单独启动后端 API 服务
python -m tools.config_studio.server

# 单独启动前端 Web 界面
cd tools/config-studio-web && npm install && npm run dev
```

## 项目架构

### 核心组件

| 组件 | 职责 |
|------|------|
| `main.py` | 启动入口，注册 Napcat 事件回调，初始化 Agent 池 |
| `agent_pool.py` | 优先级任务调度器（0-15 级），Worker 线程池，支持动态扩缩容 |
| `bot.py` | 机器人 QQ 号配置（`ncatbot_config.root`） |
| `workflows/` | 各 Agent 工作流实现，均继承 LangGraph 状态机模式 |

### 工作流设计模式

每个工作流（`workflows/summary.py`、`workflows/auto_reply.py` 等）遵循统一模式：

1. **配置加载**：`load_current_agent_config(__file__)` 获取专属配置（支持热加载）
2. **观测绑定**：`bind_agent_event()` 生成带 `run_id` 的日志函数
3. **任务提交**：`submit_agent_job()` 将任务提交至 Agent 池
4. **状态管理**：使用 LangGraph 构建有向无环图
5. **结构化输出**：利用 Pydantic 模型约束 LLM 输出格式

### 配置管理

- 配置文件：`agent_config.yaml`（YAML 格式，支持热加载）
- 各 Agent 配置节：`summary_config`、`auto_reply_config`、`forward_config`、`dida_scheduler_config`
- `agent_config_loader.py` 提供 `load_current_agent_config()` 和 `get_model_name()` 工具函数
- 模型优先级：rule.decision_model > config.decision_model > rule.model > config.model

### 优先级调度策略

- 优先级范围：0（最高）～15（最低）
- 调度策略：严格优先级抢占，相同优先级按 FIFO
- 队列容量：默认 100（可配置），满则抛出异常
- 任务超时：默认 120s，超时自动取消

### 观测日志

- 输出路径：`logs/agent_events.jsonl`
- `agent_observe.py` 提供 `observe_agent_event()` 和 `bind_agent_event()` 工具
- 日志字段：`run_id`、`agent_name`、`stage`、`latency_ms`、`decision` 等

## 目录结构

```
workflows/
  ├── __init__.py
  ├── agent_observe.py      # 统一日志观测
  ├── agent_config_loader.py # 配置加载工具
  ├── auto_reply.py         # 自动回复工作流
  ├── forward.py            # 消息转发工作流
  ├── summary.py            # 每日摘要工作流
  ├── help.py               # 帮助命令
  ├── message_observe.py    # 消息记录
  ├── dida/                 # 滴答清单集成
  │   ├── dida_agent.py
  │   ├── dida_scheduler.py
  │   └── dida_service.py
  └── scheduler/            # 定时任务调度
      ├── manager.py
      ├── registry.py
      └── actions/core.py
```
