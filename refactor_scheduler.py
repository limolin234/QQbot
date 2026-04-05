import re

with open('/home/yunzechen/Code/QQbot/tools/config-studio-web/src/App.tsx', 'r', encoding='utf-8') as f:
    text = f.read()

start_marker = "            {tab === 'scheduler' && ("
end_marker = "            {tab === 'deploy' && ("

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

scheduler_code = text[start_idx:end_idx]

# We need to extract the loop of schedules:
# It's at: {schedulerConfig.schedules.map((item, idx) => ( ... ))}
old_map_start = scheduler_code.find("{schedulerConfig.schedules.map((item, idx) => (")
old_map_end = scheduler_code.find("</ul>", old_map_start)
schedule_map_code = scheduler_code[old_map_start:old_map_end]

# Replace li with div.schedule-card
schedule_map_code = schedule_map_code.replace('<li key={idx} className={selectedScheduleIndex === idx ? \'selected\' : \'\'}>', 
    '<div key={idx} className={`schedule-card ${selectedScheduleIndex === idx ? \'selected\' : \'\'}`}>\n                                        <div className="schedule-card-header">')

# Modify the item details header
schedule_map_code = schedule_map_code.replace('<div className="row gap-8">', '<div className="row gap-8" style={{marginLeft: "auto"}}>')
schedule_map_code = schedule_map_code.replace('<strong>', '<div className="schedule-card-title">')
schedule_map_code = schedule_map_code.replace('</strong>', '</div>')
schedule_map_code = schedule_map_code.replace('<span className="muted">', '<span className="schedule-card-cron">')
schedule_map_code = schedule_map_code.replace('</li>', '</div></div>')

new_scheduler = f"""            {{tab === 'scheduler' && (
                <section className="panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
                    <div className="top-action-bar">
                        <h2 style={{ margin: 0, fontSize: '18px' }}>Scheduler</h2>
                        
                        <div style={{ width: '1px', height: '24px', background: 'var(--line)', margin: '0 12px' }}></div>
                        
                        <label style={{ margin: 0, whiteSpace: 'nowrap' }}>时区:</label>
                        <input
                            value={{schedulerConfig.timezone}}
                            onChange={{(e) => updateSchedulerConfig({{ ...schedulerConfig, timezone: e.target.value }})}}
                            style={{ margin: 0, width: '140px' }}
                        />

                        <label style={{ margin: 0, whiteSpace: 'nowrap', marginLeft: '12px' }}>条数:</label>
                        <input
                            type="number"
                            value={{timelineCount}}
                            onChange={{(e) => setTimelineCount(Number(e.target.value) || 10)}}
                            style={{ margin: 0, width: '80px' }}
                        />

                        <div className="grow"></div>
                        {{schedulerMessage && <span className="muted" style={{ marginRight: '16px' }}>{{schedulerMessage}}</span>}}

                        <button className="btn-primary" onClick={{addSchedule}}>+ 新增</button>
                        <button onClick={{saveSchedulerConfig}}>▶ 编译/应用</button>
                        <button onClick={{loadTimeline}}>刷新预览</button>
                        
                        <button 
                            className="help-icon" 
                            title="Scheduler 帮助"
                            onClick={{() => {{
                                const d = document.getElementById('help-modal-mask');
                                if(d) d.style.display = 'flex';
                            }}}}
                        >?</button>
                    </div>

                    <div id="help-modal-mask" className="modal-mask" style={{ display: 'none', zIndex: 100 }}>
                        <div className="modal-card">
                            <div className="row between">
                                <h3 style={{ margin: 0 }}>Scheduler 功能说明</h3>
                                <button onClick={{() => {{ const d = document.getElementById('help-modal-mask'); if(d) d.style.display = 'none'; }}}}>关闭</button>
                            </div>
                            <div className="scheduler-help-content" style={{ marginTop: '16px', maxHeight: '60vh', overflowY: 'auto' }}>
                                <h4>1. 可用参数</h4>
                                <ul>
                                    <li>timezone: 调度时区，例如 Asia/Shanghai。cron计算会使用该时区。</li>
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
                                    <li>保存并编译: 调用后端编译接口并保存到 agent_config.yaml。</li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    <div className="scheduler-layout">
                        <div className="scheduler-main">
                            <div className="card" style={{ height: '100%', overflowY: 'auto', border: 'none', background: 'transparent', padding: 0 }}>
                                <div className="schedule-card-list">
                                    {schedule_map_code}
                                </div>
                            </div>
                        </div>

                        <div className="scheduler-sidebar">
                            <div className="card" style={{ height: '100%', overflowY: 'auto' }}>
                                <h3>时间线预览</h3>
                                {{timeline.length === 0 && <div className="muted top-gap">点击[刷新预览]获取执行时间点</div>}}
                                <div className="timeline" style={{ maxHeight: 'none', border: 'none', padding: 0 }}>
                                    {{timeline.map((event, idx) => (
                                        <div key={{idx}} style={{ marginBottom: '8px' }}>
                                            <strong style={{ color: 'var(--accent)' }}>{{event.timestamp}}</strong>
                                            <span style={{ marginLeft: '8px' }}>{{event.name}}</span>
                                        </div>
                                    ))}}
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            )}

"""
new_text = text[:start_idx] + new_scheduler + text[end_idx:]

with open('/home/yunzechen/Code/QQbot/tools/config-studio-web/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Replaced!")
