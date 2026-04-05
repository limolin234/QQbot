import re
with open('tools/config-studio-web/src/App.tsx', 'r') as f:
    content = f.read()

# Replace h2 and convert details into a modal string
h2_pattern = re.compile(r'<h2>Scheduler 设置</h2>(?:.*?)<details className="scheduler-help top-gap" open>\s*<summary>.*?</summary>\s*<div className="scheduler-help-content">(.*?)</div>\s*</details>', re.DOTALL | re.MULTILINE)

m = h2_pattern.search(content)
if m:
    help_content = m.group(1)
    
    new_h2 = f"""<h2 style={{{{ display: 'flex', alignItems: 'center', gap: '8px' }}}}>
                            Scheduler 设置
                            <button 
                                className="icon-btn" 
                                onClick={{() => setOpenSchedulerHelp(true)}} 
                                title="使用说明" 
                                style={{{{ borderRadius: '50%', width: '24px', height: '24px', padding: 0, minWidth: 0, lineHeight: '24px', display: 'inline-block', textAlign: 'center', fontSize: '14px', cursor: 'pointer' }}}}
                            >
                                ?
                            </button>
                        </h2>
                        {{showSummaryMissingWarning && (
                            <div className="warning top-gap">
                                已开启 scheduler_manager，但未发现启用的 summary.daily_report 任务；日报不会自动执行。
                            </div>
                        )}}
                        {{showDidaPollMissingWarning && (
                            <div className="warning top-gap">
                                已配置 Dida 清单并开启 scheduler_manager，但未发现启用的 dida.poll 任务；Dida 清单不会自动拉取。
                            </div>
                        )}}
                        """
    content = content[:m.start()] + new_h2 + content[m.end():]
    
    # insert the modal at the end of section
    sec_end = content.rfind('</section>', m.end())
    modal_str = f"""<FormModal open={{openSchedulerHelp}} title="Scheduler 功能使用说明" onClose={{() => setOpenSchedulerHelp(false)}}>
                            <div className="scheduler-help-content" style={{{{ maxHeight: '600px', overflowY: 'auto' }}}}>
                                {help_content}
                            </div>
                        </FormModal>\n"""
    content = content[:sec_end] + modal_str + content[sec_end:]

# Replace the two-col section

two_col_start = content.find('<div className="two-col top-gap">')
sel_sched_start = content.find('{selectedSchedule ? (', two_col_start)
form_modal_start = content.find('<FormModal', sel_sched_start)

replacement = """
                        <div className="card top-gap">
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

                        <div className="top-gap">
                            <label>Schedule 列表</label>
                            <div className="card-grid">
                                {schedulerConfig.schedules.map((item, idx) => (
                                    <div className="rule-card" key={`${item.name}_${idx}`} style={{ display: 'flex', flexDirection: 'column' }}>
                                        <div className="row between">
                                            <strong>{item.name || `schedule_${idx + 1}`}</strong>
                                            <div className="row gap-8">
                                                <SwitchField 
                                                    checked={item.enabled} 
                                                    onChange={(enabled) => updateScheduleAt(idx, { ...item, enabled })} 
                                                    label="启用" 
                                                />
                                                <button onClick={() => { setSelectedScheduleIndex(idx); setOpenScheduleBasicModal(true); }}>编辑基础参数</button>
                                                <button onClick={() => { setSelectedScheduleIndex(idx); setOpenScheduleStepModal(true); }}>编辑执行步骤</button>
                                                <button onClick={() => duplicateSchedule(idx)}>复制</button>
                                                <button 
                                                    onClick={() => removeSchedule(idx)} 
                                                >
                                                    删除
                                                </button>
                                            </div>
                                        </div>
                                        <div className="rule-preview">
                                            类型：{item.type === 'cron' ? '定时(Cron)' : '间隔(Interval)'} | 
                                            {item.type === 'cron'
                                                ? ` 执行时间：${(() => {
                                                    const hms = hmsFromCron(item.expression);
                                                    return `${hms.hour}:${hms.minute}:${hms.second}`;
                                                })()}`
                                                : ` 执行间隔：${item.seconds || 60} 秒`}
                                        </div>
                                        <div className="rule-preview top-gap" style={{ flexGrow: 1, borderTop: '1px solid var(--border)', paddingTop: '8px' }}>
                                            <StepTreePreview nodes={item.steps_tree || []} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {selectedSchedule ? (
                            <>
"""

if two_col_start != -1 and form_modal_start != -1:
    content = content[:two_col_start] + replacement + content[form_modal_start:]

    # Need to remove the closing </div>
    # The original matched area ended with </FormModal> </div>
    # I replaced it with `... <> <FormModal ... /> <FormModal ... /> </div> : null}`
    # Wait, original ended with 
    # <FormModal .../>
    # <FormModal .../>
    # </div>
    # ) : null 
    # Since I started with `<>`, I must close with `</>`.
    
    # Let's find exactly the matching closing </div>
    content = content.replace("</div>\n                        ) : null}", "</>\n                        ) : null}")

with open('tools/config-studio-web/src/App.tsx', 'w') as f:
    f.write(content)
