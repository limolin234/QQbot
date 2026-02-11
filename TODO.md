# Refactor TODO

> 依据 `refactor.md` 当前方案（功能分治 + 统一 Agent 调度）整理。

## 已完成

- [x] 新增并可用 `agent_pool.py`（优先级队列 + worker 池 + 超时与队列满保护）。
- [x] 在启动流程接入 Agent 池初始化：`main.py` 启动时调用 `setup_agent_pool()`。
- [x] 新增通用调度入口 `submit_agent_job(...)`，支持把现有函数任务投递到池中执行。
- [x] `AUTO_REPLY` 链路先行接入调度池（`handler.py` 中由直跑改为经 `submit_agent_job(...)` 调度）。
- [x] 保持 workflows 内原有 LLM 调用方式不变（仍使用原有结构化输出路径）。

## 待完成

- [ ] 将 `summary` 链路接入 Agent 池调度（保持原 LLM 实现不变，仅改执行入口）。
- [ ] 将 `forward` 链路接入 Agent 池调度（保持原 LLM 实现不变，仅改执行入口）。
- [ ] 为 Agent 池补充运行观测（队列长度、任务耗时、超时计数、失败计数）。
- [ ] 评估并收敛配置来源（减少 `bot.py` 与 workflows 配置分散）。
- [ ] 评估 `scheduler` 中与 AI 调用强耦合部分，继续朝“功能分治”收口。

## 备注

- 当前阶段采用“先统一调度、后统一 LLM 接口”的渐进策略，优先降低重构风险。
- `refactor.md` 中“统一黑盒 LLM 调用”仍可作为后续阶段目标，不影响本阶段调度池落地。
