# Scheduler 可视化功能使用说明

## 1. 目标
本说明用于解释配置中心中 Scheduler 页当前可用的参数、操作和嵌套方式，帮助你正确配置定时任务并落盘到 workflows/agent_config.yaml。

## 2. 数据结构总览
Scheduler 配置位于：
- scheduler_manager.config.timezone
- scheduler_manager.config.schedules

每个 schedule 的核心字段：
- name: 任务名，仅用于识别和展示。
- type: 任务类型，取值 cron 或 interval。
- expression: cron 表达式，仅 type=cron 时使用。
- seconds: 间隔秒数，仅 type=interval 时使用。
- enabled: 是否启用。
- steps_tree: 步骤树，支持 action/group/if。

## 3. 页面可用操作
- 新增 Schedule：创建新的 schedule。
- 复制：复制选中 schedule，默认名称加 _copy。
- 上移/下移：调整 schedules 顺序。
- 删除：删除选中 schedule。
- 保存并编译 Scheduler：
  1) 调用 /api/scheduler/compile
  2) 将结果写回 agent_config.yaml
- 刷新时间线：调用 /api/scheduler/timeline 预览未来触发点。

## 4. 节点模型说明
### 4.1 action 节点
用于执行一个具体动作。

字段：
- kind: 固定为 action
- action: 动作 ID
- params: 参数对象

当前内置动作（前端已表单化）：
- core.send_group_msg
  - group_id: 目标群号
  - message: 消息内容
- summary.daily_report
  - run_mode: 执行模式（常见为 auto）
- dida.poll
  - project_ids: 任务项目 ID 列表
- dida.push_task_list
  - group_id: 目标群号
  - user_qq: 用户 QQ
  - day_range: 时间范围（如 today）
  - limit: 推送条数上限

说明：
- 优先使用表单编辑 params。
- 高级模式支持 JSON 直接编辑，适合扩展未内置字段。

### 4.2 group 节点
用于逻辑分组和嵌套。

字段：
- kind: 固定为 group
- name: 分组名称
- children: 子节点数组，可继续嵌套 action/group/if

用途：
- 将一段动作打包成一个块，便于阅读和复用。

### 4.3 if 节点
用于条件分支。

字段：
- kind: 固定为 if
- condition.source: 条件来源，当前支持 env 或 context
- condition.key: 条件键名
- condition.op: 比较操作符（eq/ne/contains/in）
- condition.value: 比较值
- then_steps: 条件成立时执行
- else_steps: 条件不成立时执行

示例：
- source=env
- key=ENABLE_DIDA_PUSH
- op=eq
- value=true

## 5. 嵌套方式
支持任意层级嵌套，推荐结构：
1. 顶层按业务拆为多个 schedule
2. 每个 schedule 先用 group 做阶段分组
3. 分组内部使用 if 处理可开关路径
4. 叶子节点使用 action 执行具体动作

典型结构：
- group(morning_bundle)
  - action(core.send_group_msg)
  - if(env ENABLE_DIDA_PUSH == true)
    - then: action(dida.push_task_list)
    - else: action(core.send_group_msg)

## 6. 使用建议
- 先完成 schedule 基础信息，再编辑 steps_tree。
- 先搭结构（group/if），后填 action 参数。
- 每次改完先刷新时间线，最后再保存并编译。
- 若 cron 无触发结果，优先检查 timezone 与 expression。

## 7. 常见问题
- 看不到触发时间：
  - schedule.enabled 是否为 true
  - cron 表达式是否有效
  - interval seconds 是否大于 0
- 条件分支不生效：
  - key 是否存在于 env/context
  - op 与 value 类型是否匹配
- 参数输入复杂：
  - 可先用表单，复杂对象再切换到高级 JSON 模式
