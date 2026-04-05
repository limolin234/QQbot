import re

with open('tools/config-studio-web/src/App.tsx', 'r') as f:
    content = f.read()

# 1. Remove dida_config and DidaSchedulerForm
content = content.replace("{ key: 'dida_config', label: '滴答轮询配置' },\n", "")
content = content.replace(" | 'dida_config'", "")

# remove `<DidaSchedulerForm value={didaConfig} />`
content = re.sub(r'\{\s*selectedAgent === \'dida_config\'\s*&&\s*\(\s*<DidaSchedulerForm value=\{didaConfig\}\s*\/>\s*\)\s*\}', '', content)

# delete the DidaSchedulerForm function
func_start = content.find("function DidaSchedulerForm")
func_end = content.find("function ActionParamsEditor", func_start)
content = content[:func_start] + content[func_end:]

# 2. Add openSchedulerHelp state
content = content.replace("const [tab, setTab] = useState<TabKey>('basic');", "const [tab, setTab] = useState<TabKey>('basic');\n    const [openSchedulerHelp, setOpenSchedulerHelp] = useState(false);")

# 3. Dida Form basic params optimization
dida_form_repl = """    return (
        <div>
            <div className="summary-card">
                <div className="row between wrap gap-16">
                    <div className="row gap-8" style={{ marginBottom: '16px' }}>
                        <button onClick={() => setOpenGlobalModal(true)}>编辑 Dida 全局设置</button>
                        <button onClick={() => setOpenBasicModal(true)}>编辑 Agent 参数</button>
                    </div>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '13px' }}>
                        <div className="row gap-16">
                            <span><strong>模型：</strong>{value.model || '未设置'}</span>
                            <span><strong>决策模型：</strong>{value.decision_model || '未设置'}</span>
                            <span><strong>温度：</strong>{value.temperature}</span>
                            <span><strong>机器人QQ：</strong>{value.bot_qq || '未设置'}</span>
                            <span><strong>管理员数：</strong>{value.admin_qqs.length}</span>
                        </div>
                        <div className="row gap-16">
                            <span><strong>OAuth：</strong>{value.client_id ? '已配置' : '未配置'} / {value.client_secret ? '已配置' : '未配置'}</span>
                            <span><strong>冷却：</strong>{value.min_reply_interval_seconds}s</span>
                            <span><strong>due_window：</strong>{value.due_window_seconds}s</span>
                            <span><strong>max_scan：</strong>{value.max_tasks_scan_per_user}</span>
                            <span><strong>project_ids：</strong>{value.project_ids.length}</span>
                        </div>
                    </div>
                </div>
            </div>

            <FormModal open={openGlobalModal} title="Dida 全局设置" onClose={() => setOpenGlobalModal(false)}>"""

content = re.sub(r'    return \(\n\s*<div>\n\s*<div className="summary-card">[\s\S]*?<FormModal open=\{openGlobalModal\} title="Dida 全局设置" onClose=\{...\} => setOpenGlobalModal\(false\)\}>', dida_form_repl, content)

with open('tools/config-studio-web/src/App.tsx', 'w') as f:
    f.write(content)
