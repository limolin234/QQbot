# QQBot 配置可视化前端 PRD

## 1. 文档信息
- 文档名称：QQBot 配置可视化前端 PRD
- 版本：v1.1
- 日期：2026-04-04
- 产品目标：将 .env 与 workflows/agent_config.yaml 的编辑能力从手工改文件升级为可视化配置中心，并支持一键推送到 Linux 服务器

## 2. 背景与问题
当前配置方式依赖手工编辑文本文件，存在以下问题：
- 配置门槛高：非开发者难以正确编辑 YAML 结构
- 风险高：字段名拼写错误、缩进错误、类型错误会导致运行异常
- 可观测性弱：scheduler 的执行节奏难以通过文本直观看出
- 运维链路长：本地改完后需手动执行 scp，缺乏状态反馈与审计

## 3. 目标与范围
### 3.1 核心目标
- 提供本地 Web 前端，用于可视化编辑以下配置并即时落盘：
  - .env 中的 llm_api_key、llm_base_url（兼容命名映射，见默认假设）
  - workflows/agent_config.yaml 的全部内容
- 对 agent_config.yaml 提供分区式编辑：
  - 基础 Agent 设置区（覆盖当前已有 agent 的全部配置项）
  - Scheduler 设置区（可视化、支持拖拽排序、步骤嵌套建模、时间线预览）
- 支持将本地 .env 与 agent_config.yaml 推送到 Linux 服务器

### 3.2 非目标（本期不做）
- 不做远程服务器在线编辑（即不从服务器拉取并在线改）
- 不做多人协作与权限体系
- 不做 bot 业务逻辑重构（配置中心作为工具层）

## 4. 用户角色
- 管理员（单用户）
  - 在本地编辑配置
  - 预览 scheduler 未来触发时间
  - 推送配置到服务器

## 5. 关键术语
- 配置中心：本地 Web 应用，包含前端页面与本地 API 服务
- 实时同步：前端修改后立即写入本地文件（带防抖与原子写）
- 嵌套步骤：单个 schedule 下 steps 可包含分组或子步骤结构（通过 UI 建模并编译为当前后端可执行结构）

## 6. 默认假设（由口语需求推断）
以下默认值用于避免需求歧义，已根据用户确认更新：
- A1：.env 键名使用 LLM_API_KEY 与 LLM_API_BASE_URL
- A2：配置中心默认运行在本机 127.0.0.1:4173（前端）与 127.0.0.1:8787（本地 API）
- A3：实时同步采用 300ms 防抖；每次保存先写临时文件再原子替换
- A4：scheduler 的“嵌套”与“条件分支”都需支持，编译后落地为可执行调度 DSL
- A5：时间线预览展示未来 24 小时和未来 7 天两种视图
- A6：推送功能默认目标服务器为 root@139.196.90.36，目标目录为 /root/qqbot
- A7：推送方式优先 SSH/SFTP（库方式），保留 scp 命令回退，并支持推送后自动重启

## 7. 产品方案总览
采用前后端分离的本地工具架构：
- 前端：React + Vite + TypeScript
- 本地 API：FastAPI（Python），复用现有 Python 生态与 YAML 解析能力
- 数据源：直接读写仓库下 .env 与 workflows/agent_config.yaml
- 推送通道：Paramiko SFTP（主） + scp 子进程（备）

## 8. 功能需求（FR）
### FR-1 配置文件加载
- 启动后自动读取本地 .env 与 workflows/agent_config.yaml
- 解析成功后显示表单；解析失败时显示错误位置与修复建议
- 若文件不存在：自动初始化
  - 优先使用 .env.example / workflows/agent_config.yaml.example
  - 若示例文件也不存在，使用内置默认模板自动创建并提示

验收标准：
- 能在 2 秒内完成加载并显示
- YAML 错误可定位到行号
- 在“目标文件与示例文件都不存在”的情况下，首次进入页面仍可自动生成可编辑配置

### FR-2 .env 可视化编辑
- 提供字段：
  - LLM_API_KEY（密码输入框，支持显隐）
  - LLM_API_BASE_URL（URL 输入框）
- 支持附加显示“原始键名映射”（兼容 llmapikey/llmbaseurl 历史写法）
- 修改后自动保存到本地 .env

验收标准：
- 修改后 500ms 内文件落盘
- 不破坏 .env 其他未编辑键值

### FR-3 Agent 配置编辑（agent_config.yaml）
- 按配置段展示：summary_config、forward_config、auto_reply_config、dida_agent_config、dida_config、scheduler_manager
- 每段提供：启用开关、字段表单、高级模式（原始 YAML 编辑，仅兜底）
- 表单变更实时同步 YAML；高级模式提交时做 schema 校验

验收标准：
- 表单与 YAML 双向同步
- 校验失败时阻止写入并给出错误提示
- 默认主路径不出现 JSON 文本编辑；JSON/YAML 文本编辑仅在“高级模式”中可见

### FR-4 Scheduler 可视化编排
- Schedule 列表支持：新增、复制、删除、启停、拖拽排序
- 单个 Schedule 支持：
  - 基本信息：name、type、expression 或 seconds、enabled
  - Steps 编排：动作选择、参数表单填写、拖拽排序
  - 嵌套分组：支持 Group Step（仅 UI 结构）
- 编译规则：
  - 支持 Group Step（嵌套组织）
  - 支持条件分支（if/else）
  - 生成可执行调度 DSL，并与运行时解释器一致

验收标准：
- 拖拽后 YAML 顺序实时更新
- 嵌套 UI 结构编译后可被现有 SchedulerManager 正常执行
- action 参数默认使用字段表单，非高级模式下不要求用户手写 JSON

### FR-5 时间线预览
- 对每个启用的 schedule 计算未来触发点：
  - cron：基于 timezone 与 expression 计算
  - interval：按当前时刻 + seconds 推演
- 提供两种展示：
  - 列表视图：下一次 + 后续 N 次
  - 时间线视图：24h/7d

验收标准：
- cron 表达式非法时即时提示
- 时间线显示与计算结果一致

### FR-6 实时同步与冲突处理
- 本地保存策略：
  - 防抖 300ms
  - 原子写（.tmp -> rename）
  - 保存历史快照（最近 20 份）
- 文件外部变更检测：
  - 监听文件修改时间
  - 冲突时弹窗：覆盖/合并/重新加载

验收标准：
- 高频编辑不丢数据
- 出现冲突时用户可明确选择处理策略

### FR-7 推送到服务器
- 推送对象：
  - workflows/agent_config.yaml -> /root/qqbot/workflows/agent_config.yaml
  - .env -> /root/qqbot/.env
- 支持：
  - 连接配置管理（host、port、user、auth）
  - 连接测试
  - 一键推送
  - 分文件推送
  - 推送日志与结果状态
  - 推送后自动重启（可配置）
- 安全策略：
  - 私钥优先，密码次选
  - 凭据仅存本地系统密钥环（无明文）

验收标准：
- 推送成功率可见
- 失败有明确原因（认证失败、权限不足、路径不存在等）

## 9. 非功能需求（NFR）
- 性能：
  - 初始加载 < 2s
  - 单次保存 < 500ms
- 可用性：
  - 关键操作可回滚（快照恢复）
- 安全：
  - API 服务仅绑定 127.0.0.1
  - 推送前二次确认
- 稳定性：
  - 任意解析异常不导致服务崩溃

## 10. 信息架构与界面规划
### 10.1 页面结构
- 顶部栏：项目路径、保存状态、最近保存时间、推送按钮
- 左侧导航：
  - 基础设置
  - Agent 设置
  - Scheduler 设置
  - 推送中心
  - 变更历史
- 主编辑区：对应表单与可视化画布
- 右侧辅助区：
  - 原始 YAML/.env 预览
  - 校验结果
  - 时间线预览

### 10.2 分区说明
- 基础设置：.env 关键字段（LLM_API_KEY、LLM_API_BASE_URL）
- Agent 设置：按配置段折叠面板展示
- Scheduler 设置：
  - 上半区：Schedule 列表和拖拽排序
  - 中间区：Steps 可视化编排（含嵌套分组与条件分支）
  - 下半区：时间线预览与下一次触发
- 推送中心：服务器连接、测试、推送结果、日志

## 11. 技术架构设计
### 11.1 前端
- 技术栈：React + Vite + TypeScript + Zustand
- 关键库建议：
  - 表单：react-hook-form + zod
  - 拖拽：dnd-kit
  - 时间线：vis-timeline 或 FullCalendar（timeGrid）
  - YAML 编辑器：Monaco Editor（高级模式）

### 11.2 本地 API（FastAPI）
- 模块划分：
  - config_service：读取、校验、写入 .env/.yaml
  - scheduler_compiler：UI 嵌套/条件结构 -> 调度 DSL
  - scheduler_runtime_adapter：DSL 兼容与运行时解释
  - timeline_service：cron/interval 未来触发计算
  - deploy_service：SFTP/scp 推送
  - snapshot_service：配置快照管理

### 11.3 数据模型（核心）
- EnvModel
  - llmApiKey: string
  - llmBaseUrl: string
- AgentConfigModel
  - rawSections: object
- SchedulerUiModel
  - schedules: ScheduleNode[]
- ScheduleNode
  - id, name, type, expression, seconds, enabled, steps
- StepNode
  - id, action, params, children(optional)

## 12. API 设计（本地）
- GET /api/config/all
  - 返回 .env 与 yaml 的解析结果
- PUT /api/config/env
  - 更新 .env 指定字段
- PUT /api/config/agent
  - 更新 yaml 指定 section 或整文件
- POST /api/scheduler/compile
  - 输入嵌套 UI 结构，输出线性 steps
- POST /api/scheduler/timeline
  - 输入 schedules，返回未来触发点
- POST /api/deploy/test-connection
  - 测试服务器连接
- POST /api/deploy/push
  - 推送 .env 与 yaml
- GET /api/history
  - 获取快照列表
- POST /api/history/restore
  - 回滚到指定快照

## 13. 实时同步策略
- 前端每次变更写入本地状态
- 300ms 防抖后调用保存 API
- API 进行 schema 校验与原子写
- 成功后刷新“最后保存时间”
- 失败后回滚 UI 变更并提示错误

## 14. Scheduler 嵌套与拖拽实现细则
- UI 支持三类节点：
  - Action 节点：可执行动作
  - Group 节点：逻辑分组（用于嵌套）
  - If 节点：条件分支（then/else）
- 编译阶段：
  - 将 Group/If 编译为可执行调度 DSL
  - 保持可视化结构与运行时语义一致
  - 对旧版线性 steps 保持兼容

## 15. 推送功能设计
### 15.1 连接配置
- host、port、username
- 认证方式：私钥文件路径 或 密码
- 保存策略：本地加密存储（keyring）

### 15.2 推送流程
1. 校验本地文件存在且可读
2. 建立 SSH 连接
3. 上传临时文件到服务器
4. 服务器端原子替换目标文件
5. 返回结果与日志

### 15.3 与用户提供命令的映射
- 需求等价于：
  - 上传 workflows/agent_config.yaml 到 /root/qqbot/workflows/
  - 上传 .env 到 /root/qqbot/
- 本工具默认使用 SFTP 完成，必要时可显示 scp 命令供手动执行

## 16. 启动与运行方案
### 16.1 开发模式
- 启动后端：python -m tools.config_studio.server
- 启动前端：npm run dev（目录：tools/config-studio-web）

### 16.2 生产模式（本地工具）
- 前端打包：npm run build
- 后端启动：uvicorn app:app --host 127.0.0.1 --port 8787
- 通过浏览器访问本地地址

### 16.3 项目目录建议
- tools/config_studio/server
- tools/config-studio-web
- docs/dev/config_frontend_prd.md

## 17. 里程碑计划
- M1（3-5 天）：
  - .env 编辑 + agent_config.yaml 基础表单 + 实时落盘
- M2（5-7 天）：
  - Scheduler 拖拽编排 + 时间线预览
- M3（3-4 天）：
  - 推送中心 + 连接测试 + 日志
- M4（2 天）：
  - 快照回滚 + 冲突处理 + 文档

## 18. 风险与应对
- 风险1：YAML 复杂结构导致表单映射不完整
  - 应对：保留高级原始编辑模式
- 风险2：条件分支语义与后端解释执行不一致
  - 应对：前后端共享 DSL 约束，增加 if/else 用例回归测试
- 风险3：服务器认证失败影响推送体验
  - 应对：连接测试与错误码映射
- 风险4：文件被外部改动导致覆盖冲突
  - 应对：文件监听 + 冲突弹窗 + 快照恢复

## 19. 验收用例（关键）
- UC-1：修改 LLM_API_KEY 后 .env 正确更新且其他键保持不变
- UC-2：新增一个 scheduler，拖拽调整 steps，yaml 顺序同步更新
- UC-3：输入错误 cron，页面给出阻断提示且不写入
- UC-4：时间线能展示未来 7 天触发点
- UC-5：推送成功后服务器目标文件内容与本地一致
- UC-6：推送失败时提示具体原因并保留本地改动

## 20. 待确认问题（请你确认）
1. .env 键名已确认：LLM_API_KEY 与 LLM_API_BASE_URL。
2. Scheduler 已确认：支持可视化组织与条件分支语义。
3. 推送后重启已确认：需要自动重启服务器进程。
4. 服务器认证优先级是否固定为私钥优先？是否允许只用密码？
5. 是否需要从服务器“拉取当前配置”并做本地对比后再推送？
6. 是否需要多环境（dev/staging/prod）目标服务器切换？

## 21. 结论
本 PRD 将“配置编辑、调度可视化、时间线预览、服务器推送”纳入一个本地配置中心，优先保证无歧义、可落地、可回滚。若你确认第 20 节问题，我可以进一步输出下一版：
- 交互原型草图说明
- 字段级 schema 清单
- 开发任务拆解（前端/后端/测试）

## 22. 当前待办（以表单化为主）
### 22.1 P0（必须完成）
- P0-1：文件初始化兜底
  - 在示例文件缺失时，后端自动写入内置默认模板
- P0-2：Agent 配置表单化
  - summary_config、forward_config、auto_reply_config、dida_agent_config、dida_config、scheduler_manager 全部改为字段表单
  - 当前 Section JSON 文本编辑降级为高级模式
- P0-3：Scheduler 参数表单化
  - action 参数按 action_id 渲染专属表单（如 core.send_group_msg、dida.push_task_list）
  - 当前 params JSON textarea 降级为高级模式

### 22.2 P1（应完成）
- P1-1：表单校验与错误提示完善（字段级）
- P1-2：时间线视图增强（列表 + 可视化时间轴切换）
- P1-3：推送后重启结果可视化（状态徽标 + 失败排障建议）

### 22.3 P2（可后置）
- P2-1：多环境目标管理（dev/staging/prod）
- P2-2：远程拉取并差异对比后再推送
