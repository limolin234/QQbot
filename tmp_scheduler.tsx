                                <li>timeline count: 时间线预览条数，上限建议不超过 200。</li>
                                <li>name: 任务名称，仅用于识别和展示。</li>
                                <li>type: 调度类型，可选 cron 或 interval。</li>
                                <li>执行时间(HH:MM:SS): 仅在 type=cron 时填写，系统会自动转换为 cron 表达式。</li>
                                <li>seconds: 间隔秒数，仅在 type=interval 时生效。</li>
                                <li>enabled: 是否启用该 schedule。</li>
                                <li>steps_tree: 调度步骤树，支持 action/group 两类节点。</li>
                            </ul>

                            <h4>2. 可用操作</h4>
                            <ul>
                                <li>新增 Schedule: 创建一条新的调度任务。</li>
                                <li>复制: 复制当前任务，用于快速创建相似任务。</li>
                                <li>上移/下移: 调整任务执行顺序（保存后写入 YAML 顺序）。</li>
                                <li>删除: 删除任务。</li>
                                <li>保存并编译 Scheduler: 调用后端编译接口并保存到 agent_config.yaml。</li>
                                <li>刷新时间线: 基于当前配置计算未来触发时间。</li>
                            </ul>

                            <h4>3. 节点类型与嵌套</h4>
                            <ul>
                                <li>action: 实际执行动作，包含 action 标识和 params 参数。</li>
                                <li>group: 逻辑分组节点，children 内可继续嵌套 action/group。</li>
                            </ul>

                            <h4>4. group/action 参数说明</h4>
                            <ul>
                                <li>group.name: 分组名称，仅用于结构化组织步骤。</li>
                                <li>group.children: 子步骤列表，按顺序执行。</li>
                                <li>action.action: 动作 ID，例如 core.send_group_msg、summary.daily_report、dida.poll、dida.push_task_list。</li>
                                <li>action.params: 动作参数，优先使用表单输入，高级模式可编辑 JSON。</li>
                            </ul>

                            <h4>5. 推荐使用流程</h4>
                            <ol>
                                <li>先创建 schedule 并设置 type、执行时间(HH:MM:SS)/seconds、enabled。</li>
                                <li>在 steps_tree 里先搭结构（group），再填 action 参数。</li>
                                <li>点击刷新时间线确认触发节奏。</li>
                                <li>点击保存并编译 Scheduler 落盘生效。</li>
                            </ol>
                        </div>
                    </details>

                    <div className="row gap-8 wrap">
                        <div className="grow">
                            <label>时区</label>
                            <input
                                value={schedulerConfig.timezone}
                                onChange={(e) =>
                                    updateSchedulerConfig({
                                        ...schedulerConfig,
                                        timezone: e.target.value,
                                    })
                                }
                            />
                        </div>
                        <div>
                            <label>时间线条数</label>
                            <input
                                type="number"
                                value={timelineCount}
                                onChange={(e) => setTimelineCount(Number(e.target.value) || 10)}
                            />
                        </div>
                    </div>

                    <div className="row gap-8 top-gap wrap">
                        <button onClick={addSchedule}>+ 新增 Schedule</button>
                        <button onClick={saveSchedulerConfig}>保存并编译 Scheduler</button>
                        <button onClick={loadTimeline}>刷新时间线</button>
                        {schedulerMessage && <span className="muted">{schedulerMessage}</span>}
                    </div>

                    <div className="two-col top-gap">
                        <div className="card">
                            <h3>Schedule 列表</h3>
                            <ul className="list">
                                {schedulerConfig.schedules.map((item, idx) => (
                                    <li
                                        key={`${item.name}_${idx}`}
                                        className={selectedScheduleIndex === idx ? 'selected' : ''}
                                        onClick={() => setSelectedScheduleIndex(idx)}
                                    >
                                        <div className="grow">
                                            <div>
                                                {item.name || `schedule_${idx + 1}`} | {item.type === 'cron' ? '定时(Cron)' : '间隔(Interval)'} | {item.enabled ? '已启用' : '已停用'}
                                            </div>
                                            <div className="rule-preview">
                                                {item.type === 'cron'
                                                    ? `执行时间: ${(() => {
                                                        const hms = hmsFromCron(item.expression);
                                                        return `${hms.hour}:${hms.minute}:${hms.second}`;
                                                    })()}`
                                                    : `执行间隔: ${item.seconds || 60} 秒`}
                                            </div>
                                            <StepTreePreview nodes={item.steps_tree || []} />
                                        </div>
                                        <div className="row gap-8 wrap">
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    moveSchedule(idx, -1);
                                                }}
                                                disabled={idx === 0}
                                            >
                                                上移
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    moveSchedule(idx, 1);
                                                }}
                                                disabled={idx === schedulerConfig.schedules.length - 1}
                                            >
                                                下移
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    duplicateSchedule(idx);
                                                }}
                                            >
                                                复制
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    removeSchedule(idx);
                                                }}
                                            >
                                                删除
                                            </button>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div className="card">
                            <h3>时间线预览</h3>
                            <div className="timeline">
                                {timelineEvents.length === 0 ? (
                                    <div className="muted">暂无数据，点击“刷新时间线”生成。</div>
                                ) : (
                                    timelineEvents.map((event, idx) => (
                                        <div key={`${event.schedule_name}_${event.trigger_at}_${idx}`}>
                                            [{event.source}] {event.schedule_name} - {event.trigger_at}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>
