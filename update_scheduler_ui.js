const fs = require('fs');
let code = fs.readFileSync('tools/config-studio-web/src/App.tsx', 'utf-8');

const startMarker = '<div className="two-col top-gap">';
const endMarker = '<FormModal';

const idxStart = code.lastIndexOf(startMarker, code.indexOf('<h3>当前任务</h3>'));
// Find the first FormModal inside the selectedSchedule block
const idxSelectedSchedule = code.indexOf('{selectedSchedule ? (');
const idxEnd = code.indexOf(endMarker, idxSelectedSchedule);

if (idxStart === -1 || idxEnd === -1) {
    console.error("Markers not found");
    process.exit(1);
}

const replacement = `
                        <div className="card top-gap">
                            <h3>时间线预览</h3>
                            <div className="timeline">
                                {timelineEvents.length === 0 ? (
                                    <div className="muted">暂无数据，点击“刷新时间线”生成。</div>
                                ) : (
                                    timelineEvents.map((event, idx) => (
                                        <div key={\`\${event.schedule_name}_\${event.trigger_at}_\${idx}\`}>
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
                                    <div className="rule-card" key={\`\${item.name}_\${idx}\`} style={{ display: 'flex', flexDirection: 'column' }}>
                                        <div className="row between">
                                            <strong>{item.name || \`schedule_\${idx + 1}\`}</strong>
                                            <div className="row gap-8">
                                                <SwitchField 
                                                    checked={item.enabled} 
                                                    onChange={(enabled) => updateScheduleAt(idx, { ...item, enabled })} 
                                                    label="启用" 
                                                />
                                                <button onClick={() => { setSelectedScheduleIndex(idx); setOpenScheduleBasicModal(true); }}>编辑基础参数</button>
                                                <button onClick={() => { setSelectedScheduleIndex(idx); setOpenScheduleStepModal(true); }}>编辑执行步骤</button>
                                                <button onClick={() => duplicateSchedule(idx)}>复制</button>
                                                <button onClick={() => removeSchedule(idx)}>删除</button>
                                            </div>
                                        </div>
                                        <div className="rule-preview">
                                            类型：{item.type === 'cron' ? '定时(Cron)' : '间隔(Interval)'} | 
                                            {item.type === 'cron'
                                                ? \` 执行时间：\${(() => {
                                                    const hms = hmsFromCron(item.expression);
                                                    return \`\${hms.hour}:\${hms.minute}:\${hms.second}\`;
                                                })()}\`
                                                : \` 执行间隔：\${item.seconds || 60} 秒\`}
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
                                `;

code = code.substring(0, idxStart) + replacement + code.substring(idxEnd);

// Because we replaced `<FormModal`, we must not absorb it, or we add it back.
// I added `<FormModal` into endMarker, so it's excluded from replacement. Wait, my replacement ends with `>`. 
// So `code.substring(idxEnd)` starts with `<FormModal`. This is correct.

// Wait, the original code had:
// {selectedSchedule ? (
//    <div className="card top-gap">
//         <h3>当前任务</h3> ...
//      <FormModal ...

// By doing this, we also need to make sure the closing `</div>` for `<div className="card top-gap">` is removed, but it's AFTER the two Modals.
fs.writeFileSync('update_app2.js', `console.log("Ready to fix closing tags")`);

fs.writeFileSync('tools/config-studio-web/src/App.tsx', code);
