# QQBot 开发 TODO（基于 `agent_requests.md`）

## P0 必做（先跑起来）

- [x] 配置真实参数：填写 `QQnumber`、`allowed_id`、关键词集合（`bot.py`）
- [ ] 统一群号类型（`str/int` 一致），避免 `allowed_id` 过滤失效（`bot.py`、`scheduler.py`）
- [ ] 实现 `SUMMARY` 任务：对日志片段做总结并私发给自己（`handler.py`）
- [ ] 实现 `FORWARD` 任务：先判断“是否值得转发”，再执行转发/跳过（`handler.py`）
- [ ] 实现 `GROUPNOTE`（URGENT）任务：判断是否接龙、生成内容、发群并私发通知（`handler.py`）

## P1 安全与稳定（强烈建议）

- [ ] 给 URGENT 分支加发送频率限制（例如每群每分钟上限）（`handler.py`）
- [ ] 给群发内容加长度上限与关键词黑名单，超限直接拒绝（`handler.py`）
- [ ] 失败兜底：发送失败重试 1 次并记录错误日志（`handler.py`）
- [ ] 补充关键日志：任务类型、来源群、是否发送、失败原因（`handler.py`）

## P2 体验优化（可选）

- [ ] 启动多个 worker 并行处理（先 2~3 个），观察是否有冲突（`main.py`）
- [ ] 将策略参数（频率阈值、长度上限）从硬编码提取到配置（`bot.py`）
- [ ] 统一命名：`FROWARD` 更正为 `FORWARD`（全项目一致替换，含 `scheduler.py`、`handler.py`）

## 验收标准（完成即通过）

- [ ] 群消息命中关键词后，`worker` 能正确消费并进入对应分支（`main.py`）
- [ ] 每晚 22:00 的 `SUMMARY` 能正常生成并私发（`main.py`、`scheduler.py`）
- [ ] URGENT 场景触发安全限制时，不会发生群发失控

## 备注

- 任务来源：`agent_requests.md`
- 建议按优先级从 `P0` 到 `P2` 顺序推进。
