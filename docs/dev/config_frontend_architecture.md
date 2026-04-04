# 配置可视化中心 架构方案 (Architect v1)

## 0. 结论摘要
目标是以最小侵入方式，为 QQBot 增加本地配置中心（前端 + 本地 API），实现：
- 可视化编辑 .env 与 workflows/agent_config.yaml 并实时落盘
- Scheduler 支持拖拽、嵌套、条件分支，并具备时间线预览
- 推送配置到 Linux 服务器并自动重启进程

策略：
- 业务 Bot 主链路尽量不改；新增 tools/config_studio 工具层承载前后端
- 仅在 workflows/scheduler 增量支持“调度 DSL 解释器”
- 保留现有线性 steps 兼容，平滑升级

## 1. 上下文剪裁 (Context Pruning)
### 1.1 Coder 必读文件（只读/改动目标）
- main.py
- workflows/agent_config_loader.py
- workflows/scheduler/config_models.py
- workflows/scheduler/manager.py
- workflows/scheduler/actions/core.py
- workflows/agent_config.yaml.example
- deploy/docker-compose.yml
- .env.example

### 1.2 本次新增目录（主要开发面）
- tools/config_studio/server/
- tools/config-studio-web/
- docs/dev/config_frontend_architecture.md

### 1.3 明确不要改的区域（降低风险）
- workflows/summary.py（除 action 参数映射问题外不改实现）
- workflows/forward.py
- workflows/auto_reply.py
- workflows/dida/dida_agent.py
- agent_pool.py

## 2. 影响范围分析
- 新增：tools/config_studio/server/app.py
  - 本地 API 入口
- 新增：tools/config_studio/server/services/config_service.py
  - .env / YAML 读取、校验、原子写
- 新增：tools/config_studio/server/services/timeline_service.py
  - cron / interval 未来触发点计算
- 新增：tools/config_studio/server/services/deploy_service.py
  - SFTP 推送与远程重启
- 新增：tools/config_studio/server/services/snapshot_service.py
  - 快照与回滚
- 新增：tools/config_studio/server/models/*.py
  - API 请求响应模型
- 新增：tools/config-studio-web/*
  - 前端页面与状态管理
- 修改：workflows/scheduler/config_models.py
  - 增加条件分支 DSL 模型（兼容原 steps）
- 修改：workflows/scheduler/manager.py
  - 增加 DSL 解释执行器（if/else + group）
- 修改：workflows/agent_config.yaml.example
  - 增加条件分支与嵌套示例
- 修改：requirements.txt
  - 增加 fastapi、uvicorn、paramiko、croniter、watchfiles、python-dotenv（如未显式）

## 3. 依赖分析 (Dependency Analysis)
新增依赖与兼容性：
- fastapi + uvicorn
  - 与现有 asyncio 体系兼容；仅用于本地工具服务
- paramiko
  - 用于 SFTP 与 SSH 远程重启；不影响 bot 运行时
- croniter
  - 用于时间线预览；与 aiocron 并存无冲突
- watchfiles
  - 文件外部变更检测；仅配置中心使用
- pyyaml 已存在，可复用

风险控制：
- 将新依赖限定在 tools/config_studio 路径调用，不注入主业务链路
- 所有推送/重启 API 默认仅绑定 127.0.0.1

## 4. 数据流设计 (Data Flow Design)
### 4.1 本地编辑流
1. 前端读取 /api/config/all
2. 用户编辑表单
3. 前端 300ms 防抖提交 PUT /api/config/env 或 PUT /api/config/agent
4. 后端校验 -> 原子写 -> 生成快照
5. 返回保存版本号与时间戳，前端更新状态

### 4.2 Scheduler 可视化流
1. 前端维护树状编辑模型（group / if / action）
2. 提交 POST /api/scheduler/compile
3. 后端编译为调度 DSL（可执行结构）并返回
4. 写回 agent_config.yaml 的 scheduler_manager.config.schedules
5. POST /api/scheduler/timeline 生成未来触发点

### 4.3 推送与自动重启流
1. 前端点击推送
2. 后端执行：连接测试 -> 上传临时文件 -> 原子替换
3. 上传成功后执行重启命令
4. 返回完整日志（上传结果、重启结果、stderr）

## 5. 接口定义
### 5.1 本地 API 伪代码
```python
# GET /api/config/all
async def get_all_config() -> AllConfigResponse

# PUT /api/config/env
async def update_env(payload: EnvPatchRequest) -> SaveResult

# PUT /api/config/agent
async def update_agent_yaml(payload: AgentPatchRequest) -> SaveResult

# POST /api/scheduler/compile
async def compile_scheduler(payload: SchedulerUiPayload) -> SchedulerDslPayload

# POST /api/scheduler/timeline
async def preview_timeline(payload: TimelineRequest) -> TimelineResponse

# POST /api/deploy/test-connection
async def test_connection(payload: DeployTarget) -> ConnectionResult

# POST /api/deploy/push
async def push_and_restart(payload: DeployPushRequest) -> DeployResult
```

### 5.2 核心类型（伪代码）
```python
class EnvPatchRequest(BaseModel):
    llm_api_key: str
    llm_api_base_url: str

class ActionNode(BaseModel):
    kind: Literal["action"]
    action: str
    params: dict[str, Any] = {}

class GroupNode(BaseModel):
    kind: Literal["group"]
    name: str
    children: list[StepNode]

class IfNode(BaseModel):
    kind: Literal["if"]
    condition: dict[str, Any]  # 例如 {"source": "env", "key": "ENABLE_DIDA", "op": "eq", "value": "true"}
    then_steps: list[StepNode]
    else_steps: list[StepNode] = []

StepNode = ActionNode | GroupNode | IfNode

class ScheduleDsl(BaseModel):
    name: str
    type: Literal["cron", "interval"]
    expression: str | None = None
    seconds: int | None = None
    enabled: bool = True
    steps_tree: list[StepNode] = []
```

### 5.3 Scheduler 运行时执行接口（伪代码）
```python
async def execute_steps_tree(steps_tree: list[StepNode], context: dict[str, Any]) -> None

async def execute_action(step: ActionNode, context: dict[str, Any]) -> None

def eval_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool
```

## 6. UI 规划
- 顶部：项目路径、保存状态、推送按钮、重启状态
- 左侧导航：基础设置 / Agent 设置 / Scheduler / 推送中心 / 历史快照
- 中央：
  - 基础设置：LLM_API_KEY、LLM_API_BASE_URL
  - Agent 设置：按 section 折叠表单
  - Scheduler：
    - 左栏 schedule 列表（拖拽）
    - 中栏 step 树编辑器（action/group/if）
    - 右栏参数面板与校验
- 底部：时间线（24h/7d）与下一次触发

## 7. 自动重启策略
重启优先顺序（可配置）：
1. docker compose restart qqbot
2. systemctl restart qqbot
3. pm2 restart qqbot

默认实现：优先执行 docker compose（工作目录 /root/qqbot）
- 远程命令：cd /root/qqbot && docker compose up -d qqbot
- 若失败则回退：cd /root/qqbot && docker compose restart qqbot

## 8. 最小侵入改造原则
- Bot 主入口 main.py 不引入前端工具依赖
- 配置中心独立运行，不影响 bot 正常启动
- Scheduler 增强保持向后兼容：
  - 若只有旧版 steps，按旧逻辑执行
  - 若存在 steps_tree，走新解释器

## 9. 实施步骤 (Task List)
1. [Step 1] 初始化 tools/config_studio/server FastAPI 工程与基础健康检查接口。
2. [Step 2] 实现 config_service：.env/.yaml 读取、schema 校验、原子写、快照。
3. [Step 3] 实现前端基础页面与 .env 编辑（实时保存）。
4. [Step 4] 实现 Agent 设置表单（section 级编辑 + 原始 YAML 高级模式）。
5. [Step 5] 实现 Scheduler 可视化编辑器（拖拽 + group + if 节点）。
6. [Step 6] 在 workflows/scheduler/config_models.py 增加 steps_tree 数据结构（兼容旧 steps）。
7. [Step 7] 在 workflows/scheduler/manager.py 增加 execute_steps_tree 解释器与条件执行。
8. [Step 8] 实现 timeline_service（croniter + interval 预测）。
9. [Step 9] 实现 deploy_service：SFTP 上传 + 原子替换 + 自动重启。
10. [Step 10] 增加推送中心 UI（连接测试、推送日志、失败回显）。
11. [Step 11] 更新 workflows/agent_config.yaml.example 与 scheduler 示例，加入 if/group 示例。
12. [Step 12] 联调与验收：本地编辑、时间线、推送、重启全链路测试。

## 10. 验收基线
- 编辑 .env 后 500ms 内落盘，键名严格为 LLM_API_KEY / LLM_API_BASE_URL
- Scheduler 支持 if/else 与 group，执行结果与 UI 一致
- 推送后自动重启成功并返回日志
- 任何失败均可回滚快照并恢复
