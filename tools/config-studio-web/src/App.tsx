import { useEffect, useMemo, useState } from 'react';
import {
    compileSchedules,
    deployPush,
    getAgentRaw,
    getAllConfig,
    getTimeline,
    listSnapshots,
    restoreSnapshot,
    saveAgent,
    saveAgentRaw,
    saveEnv,
    testConnection,
} from './api';
import type { AgentConfigRoot, ScheduleConfig, StepNode, TabKey, TimelineEvent } from './types';

function newId() {
    return Math.random().toString(36).slice(2, 10);
}

function ensureScheduler(agent: AgentConfigRoot): { timezone: string; schedules: ScheduleConfig[] } {
    const root = agent.scheduler_manager as
        | {
            file_name?: string;
            config?: { timezone?: string; schedules?: ScheduleConfig[] };
        }
        | undefined;

    if (!root) {
        agent.scheduler_manager = {
            file_name: 'scheduler_manager.py',
            config: { timezone: 'Asia/Shanghai', schedules: [] },
        };
    }

    const cfg = (agent.scheduler_manager as { config: { timezone?: string; schedules?: ScheduleConfig[] } }).config;
    if (!cfg.timezone) cfg.timezone = 'Asia/Shanghai';
    if (!cfg.schedules) cfg.schedules = [];

    return {
        timezone: cfg.timezone,
        schedules: cfg.schedules,
    };
}

function normalizeStepsTree(nodes?: StepNode[]): StepNode[] {
    if (!nodes) return [];
    return nodes.map((node) => {
        const base: StepNode = {
            id: node.id || newId(),
            kind: node.kind,
            action: node.action || '',
            params: node.params || {},
            name: node.name || '',
            condition: node.condition || { source: 'env', key: '', op: 'eq', value: '' },
            children: normalizeStepsTree(node.children),
            then_steps: normalizeStepsTree(node.then_steps),
            else_steps: normalizeStepsTree(node.else_steps),
        };
        return base;
    });
}

function NodeEditor({
    nodes,
    onChange,
    title,
}: {
    nodes: StepNode[];
    onChange: (nodes: StepNode[]) => void;
    title: string;
}) {
    const [dragIndex, setDragIndex] = useState<number | null>(null);

    const addNode = (kind: StepNode['kind']) => {
        onChange([
            ...nodes,
            {
                id: newId(),
                kind,
                action: '',
                params: {},
                name: '',
                condition: { source: 'env', key: '', op: 'eq', value: '' },
                children: [],
                then_steps: [],
                else_steps: [],
            },
        ]);
    };

    const updateNode = (idx: number, patch: Partial<StepNode>) => {
        const next = [...nodes];
        next[idx] = { ...next[idx], ...patch };
        onChange(next);
    };

    const removeNode = (idx: number) => {
        const next = [...nodes];
        next.splice(idx, 1);
        onChange(next);
    };

    const onDrop = (idx: number) => {
        if (dragIndex === null || dragIndex === idx) return;
        const next = [...nodes];
        const [item] = next.splice(dragIndex, 1);
        next.splice(idx, 0, item);
        onChange(next);
        setDragIndex(null);
    };

    return (
        <div className="card">
            <div className="row between">
                <strong>{title}</strong>
                <div className="row gap-8">
                    <button onClick={() => addNode('action')}>+ Action</button>
                    <button onClick={() => addNode('group')}>+ Group</button>
                    <button onClick={() => addNode('if')}>+ If</button>
                </div>
            </div>
            {nodes.map((node, idx) => (
                <div
                    key={node.id}
                    className="node"
                    draggable
                    onDragStart={() => setDragIndex(idx)}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={() => onDrop(idx)}
                >
                    <div className="row between">
                        <strong>{node.kind.toUpperCase()}</strong>
                        <button onClick={() => removeNode(idx)}>删除</button>
                    </div>
                    {node.kind === 'action' && (
                        <>
                            <input
                                placeholder="action id, e.g. core.send_group_msg"
                                value={node.action || ''}
                                onChange={(e) => updateNode(idx, { action: e.target.value })}
                            />
                            <textarea
                                rows={4}
                                value={JSON.stringify(node.params || {}, null, 2)}
                                onChange={(e) => {
                                    try {
                                        updateNode(idx, { params: JSON.parse(e.target.value) });
                                    } catch {
                                        // Ignore parse errors while typing.
                                    }
                                }}
                            />
                        </>
                    )}
                    {node.kind === 'group' && (
                        <>
                            <input
                                placeholder="group name"
                                value={node.name || ''}
                                onChange={(e) => updateNode(idx, { name: e.target.value })}
                            />
                            <NodeEditor
                                title="Group Children"
                                nodes={node.children || []}
                                onChange={(children) => updateNode(idx, { children })}
                            />
                        </>
                    )}
                    {node.kind === 'if' && (
                        <>
                            <div className="grid4">
                                <input
                                    placeholder="source: env/context"
                                    value={node.condition?.source || ''}
                                    onChange={(e) =>
                                        updateNode(idx, {
                                            condition: { ...(node.condition || {}), source: e.target.value },
                                        })
                                    }
                                />
                                <input
                                    placeholder="key"
                                    value={node.condition?.key || ''}
                                    onChange={(e) =>
                                        updateNode(idx, {
                                            condition: { ...(node.condition || {}), key: e.target.value },
                                        })
                                    }
                                />
                                <input
                                    placeholder="op(eq/ne/contains/in/truthy/falsy)"
                                    value={node.condition?.op || ''}
                                    onChange={(e) =>
                                        updateNode(idx, {
                                            condition: { ...(node.condition || {}), op: e.target.value },
                                        })
                                    }
                                />
                                <input
                                    placeholder="value"
                                    value={node.condition?.value || ''}
                                    onChange={(e) =>
                                        updateNode(idx, {
                                            condition: { ...(node.condition || {}), value: e.target.value },
                                        })
                                    }
                                />
                            </div>
                            <NodeEditor
                                title="THEN"
                                nodes={node.then_steps || []}
                                onChange={(then_steps) => updateNode(idx, { then_steps })}
                            />
                            <NodeEditor
                                title="ELSE"
                                nodes={node.else_steps || []}
                                onChange={(else_steps) => updateNode(idx, { else_steps })}
                            />
                        </>
                    )}
                </div>
            ))}
        </div>
    );
}

export default function App() {
    const [tab, setTab] = useState<TabKey>('basic');
    const [agent, setAgent] = useState<AgentConfigRoot>({});
    const [rawYaml, setRawYaml] = useState('');
    const [status, setStatus] = useState('idle');
    const [error, setError] = useState('');

    const [llmApiKey, setLlmApiKey] = useState('');
    const [llmApiBaseUrl, setLlmApiBaseUrl] = useState('');

    const [activeSection, setActiveSection] = useState('');
    const [sectionText, setSectionText] = useState('{}');

    const [selectedSchedule, setSelectedSchedule] = useState(0);
    const [timeline, setTimeline] = useState<TimelineEvent[]>([]);

    const [deployForm, setDeployForm] = useState({
        host: '139.196.90.36',
        port: 22,
        username: 'root',
        auth_type: 'password',
        password: '',
        key_path: '',
        project_dir: '/root/qqbot',
        push_env: true,
        push_agent_yaml: true,
        restart_policy: 'docker-compose',
    });
    const [deployLogs, setDeployLogs] = useState<string[]>([]);
    const [snapshots, setSnapshots] = useState<string[]>([]);

    const scheduler = useMemo(() => ensureScheduler(agent), [agent]);
    const schedules = scheduler.schedules;

    useEffect(() => {
        (async () => {
            try {
                setStatus('loading');
                const all = await getAllConfig();
                const raw = await getAgentRaw();
                setAgent(all.agent || {});
                setRawYaml(raw.raw_yaml || '');
                setLlmApiKey(all.env?.LLM_API_KEY || '');
                setLlmApiBaseUrl(all.env?.LLM_API_BASE_URL || '');
                const sections = Object.keys(all.agent || {});
                if (sections.length > 0) {
                    setActiveSection(sections[0]);
                    setSectionText(JSON.stringify((all.agent as Record<string, unknown>)[sections[0]], null, 2));
                }
                const history = await listSnapshots();
                setSnapshots(history.snapshots || []);
                setStatus('ready');
            } catch (e) {
                setError(String(e));
                setStatus('error');
            }
        })();
    }, []);

    useEffect(() => {
        const id = setTimeout(() => {
            if (status !== 'ready') return;
            saveEnv(llmApiKey, llmApiBaseUrl).catch((e) => setError(String(e)));
        }, 300);
        return () => clearTimeout(id);
    }, [llmApiKey, llmApiBaseUrl, status]);

    const updateAgent = (next: AgentConfigRoot) => {
        setAgent({ ...next });
    };

    const saveAgentFull = async () => {
        try {
            await saveAgent(agent as Record<string, unknown>);
            setStatus('saved');
        } catch (e) {
            setError(String(e));
        }
    };

    const saveRaw = async () => {
        try {
            await saveAgentRaw(rawYaml);
            const all = await getAllConfig();
            setAgent(all.agent || {});
            setStatus('saved');
        } catch (e) {
            setError(String(e));
        }
    };

    const changeSection = (name: string) => {
        setActiveSection(name);
        setSectionText(JSON.stringify((agent as Record<string, unknown>)[name], null, 2));
    };

    const saveSection = async () => {
        try {
            const parsed = JSON.parse(sectionText);
            const next = { ...(agent as Record<string, unknown>), [activeSection]: parsed };
            updateAgent(next as AgentConfigRoot);
            await saveAgent(next);
            setStatus('saved');
        } catch (e) {
            setError(`Section JSON invalid: ${String(e)}`);
        }
    };

    const updateSchedules = (nextSchedules: ScheduleConfig[]) => {
        const next = { ...(agent as Record<string, unknown>) };
        const schedulerManager = (next.scheduler_manager as Record<string, unknown>) || {
            file_name: 'scheduler_manager.py',
            config: {},
        };
        const config = (schedulerManager.config as Record<string, unknown>) || {};
        config.timezone = scheduler.timezone;
        config.schedules = nextSchedules;
        schedulerManager.config = config;
        next.scheduler_manager = schedulerManager;
        updateAgent(next as AgentConfigRoot);
    };

    const updateTimezone = (timezone: string) => {
        const next = { ...(agent as Record<string, unknown>) };
        const schedulerManager = (next.scheduler_manager as Record<string, unknown>) || {
            file_name: 'scheduler_manager.py',
            config: {},
        };
        const config = (schedulerManager.config as Record<string, unknown>) || {};
        config.timezone = timezone;
        config.schedules = schedules;
        schedulerManager.config = config;
        next.scheduler_manager = schedulerManager;
        updateAgent(next as AgentConfigRoot);
    };

    const addSchedule = () => {
        updateSchedules([
            ...schedules,
            {
                name: `schedule_${schedules.length + 1}`,
                type: 'cron',
                expression: '0 8 * * *',
                enabled: true,
                steps_tree: [],
                steps: [],
            },
        ]);
        setSelectedSchedule(schedules.length);
    };

    const saveScheduler = async () => {
        try {
            const compiled = await compileSchedules(schedules);
            updateSchedules(compiled.schedules);
            await saveAgent(agent as Record<string, unknown>);
            setStatus('saved');
        } catch (e) {
            setError(String(e));
        }
    };

    const loadTimeline = async () => {
        try {
            const ret = await getTimeline(scheduler.timezone, schedules, 20);
            setTimeline(ret.events);
        } catch (e) {
            setError(String(e));
        }
    };

    const currentSchedule = schedules[selectedSchedule];

    const runConnectionTest = async () => {
        try {
            const result = await testConnection(deployForm);
            setDeployLogs(result.logs || [result.message]);
        } catch (e) {
            setError(String(e));
        }
    };

    const runDeploy = async () => {
        try {
            const result = await deployPush(deployForm);
            setDeployLogs(result.logs || [result.message]);
        } catch (e) {
            setError(String(e));
        }
    };

    const reloadHistory = async () => {
        const history = await listSnapshots();
        setSnapshots(history.snapshots || []);
    };

    const restore = async (snapshot: string, target: 'env' | 'agent') => {
        try {
            await restoreSnapshot(snapshot, target);
            await reloadHistory();
            setStatus('restored');
        } catch (e) {
            setError(String(e));
        }
    };

    return (
        <div className="layout">
            <header className="topbar">
                <h1>QQBot Config Studio</h1>
                <div className="status">状态: {status}</div>
            </header>

            <nav className="tabs">
                {[
                    ['basic', '基础设置'],
                    ['agent', 'Agent设置'],
                    ['scheduler', 'Scheduler设置'],
                    ['deploy', '推送中心'],
                    ['history', '变更历史'],
                ].map(([key, label]) => (
                    <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key as TabKey)}>
                        {label}
                    </button>
                ))}
            </nav>

            {error && <div className="error">{error}</div>}

            {tab === 'basic' && (
                <section className="panel">
                    <h2>基础设置 (.env)</h2>
                    <label>LLM_API_KEY</label>
                    <input value={llmApiKey} onChange={(e) => setLlmApiKey(e.target.value)} />
                    <label>LLM_API_BASE_URL</label>
                    <input value={llmApiBaseUrl} onChange={(e) => setLlmApiBaseUrl(e.target.value)} />
                    <p>说明：此页输入会 300ms 防抖自动保存。</p>
                </section>
            )}

            {tab === 'agent' && (
                <section className="panel two-col">
                    <div className="card">
                        <h3>Section 编辑</h3>
                        <select value={activeSection} onChange={(e) => changeSection(e.target.value)}>
                            {Object.keys(agent).map((k) => (
                                <option key={k} value={k}>
                                    {k}
                                </option>
                            ))}
                        </select>
                        <textarea rows={18} value={sectionText} onChange={(e) => setSectionText(e.target.value)} />
                        <div className="row gap-8">
                            <button onClick={saveSection}>保存当前 Section</button>
                            <button onClick={saveAgentFull}>保存全部</button>
                        </div>
                    </div>
                    <div className="card">
                        <h3>高级模式（原始 YAML）</h3>
                        <textarea rows={22} value={rawYaml} onChange={(e) => setRawYaml(e.target.value)} />
                        <button onClick={saveRaw}>保存原始 YAML</button>
                    </div>
                </section>
            )}

            {tab === 'scheduler' && (
                <section className="panel two-col">
                    <div className="card">
                        <div className="row between">
                            <h3>Schedules（可拖动排序）</h3>
                            <button onClick={addSchedule}>新增 Schedule</button>
                        </div>
                        <label>Timezone</label>
                        <input value={scheduler.timezone} onChange={(e) => updateTimezone(e.target.value)} />
                        <ul className="list">
                            {schedules.map((s, idx) => (
                                <li key={`${s.name}_${idx}`} className={selectedSchedule === idx ? 'selected' : ''}>
                                    <button onClick={() => setSelectedSchedule(idx)}>{s.name}</button>
                                    <button
                                        onClick={() => {
                                            if (idx === 0) return;
                                            const next = [...schedules];
                                            const [item] = next.splice(idx, 1);
                                            next.splice(idx - 1, 0, item);
                                            updateSchedules(next);
                                            setSelectedSchedule(idx - 1);
                                        }}
                                    >
                                        ↑
                                    </button>
                                    <button
                                        onClick={() => {
                                            if (idx === schedules.length - 1) return;
                                            const next = [...schedules];
                                            const [item] = next.splice(idx, 1);
                                            next.splice(idx + 1, 0, item);
                                            updateSchedules(next);
                                            setSelectedSchedule(idx + 1);
                                        }}
                                    >
                                        ↓
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className="card">
                        {currentSchedule ? (
                            <>
                                <h3>Schedule 编辑</h3>
                                <label>Name</label>
                                <input
                                    value={currentSchedule.name}
                                    onChange={(e) => {
                                        const next = [...schedules];
                                        next[selectedSchedule] = { ...currentSchedule, name: e.target.value };
                                        updateSchedules(next);
                                    }}
                                />
                                <label>Type</label>
                                <select
                                    value={currentSchedule.type}
                                    onChange={(e) => {
                                        const type = e.target.value as 'cron' | 'interval';
                                        const next = [...schedules];
                                        next[selectedSchedule] = { ...currentSchedule, type };
                                        updateSchedules(next);
                                    }}
                                >
                                    <option value="cron">cron</option>
                                    <option value="interval">interval</option>
                                </select>
                                {currentSchedule.type === 'cron' ? (
                                    <>
                                        <label>Expression</label>
                                        <input
                                            value={currentSchedule.expression || ''}
                                            onChange={(e) => {
                                                const next = [...schedules];
                                                next[selectedSchedule] = { ...currentSchedule, expression: e.target.value };
                                                updateSchedules(next);
                                            }}
                                        />
                                    </>
                                ) : (
                                    <>
                                        <label>Seconds</label>
                                        <input
                                            type="number"
                                            value={currentSchedule.seconds || 60}
                                            onChange={(e) => {
                                                const next = [...schedules];
                                                next[selectedSchedule] = { ...currentSchedule, seconds: Number(e.target.value) };
                                                updateSchedules(next);
                                            }}
                                        />
                                    </>
                                )}

                                <NodeEditor
                                    title="Steps Tree"
                                    nodes={normalizeStepsTree(currentSchedule.steps_tree || [])}
                                    onChange={(steps_tree) => {
                                        const next = [...schedules];
                                        next[selectedSchedule] = { ...currentSchedule, steps_tree };
                                        updateSchedules(next);
                                    }}
                                />

                                <div className="row gap-8">
                                    <button onClick={saveScheduler}>编译并保存</button>
                                    <button onClick={loadTimeline}>刷新时间线</button>
                                </div>
                                <div className="timeline">
                                    {timeline.map((event, idx) => (
                                        <div key={`${event.schedule_name}_${idx}`}>{event.trigger_at} | {event.schedule_name} | {event.source}</div>
                                    ))}
                                </div>
                            </>
                        ) : (
                            <p>暂无 Schedule，请先新增。</p>
                        )}
                    </div>
                </section>
            )}

            {tab === 'deploy' && (
                <section className="panel two-col">
                    <div className="card">
                        <h3>推送配置</h3>
                        {Object.entries(deployForm).map(([k, v]) => {
                            if (typeof v === 'boolean') {
                                return (
                                    <label key={k} className="row gap-8">
                                        <input
                                            type="checkbox"
                                            checked={v}
                                            onChange={(e) => setDeployForm({ ...deployForm, [k]: e.target.checked })}
                                        />
                                        {k}
                                    </label>
                                );
                            }
                            return (
                                <div key={k}>
                                    <label>{k}</label>
                                    <input
                                        value={String(v)}
                                        onChange={(e) =>
                                            setDeployForm({
                                                ...deployForm,
                                                [k]: k === 'port' ? Number(e.target.value) : e.target.value,
                                            })
                                        }
                                    />
                                </div>
                            );
                        })}
                        <div className="row gap-8">
                            <button onClick={runConnectionTest}>测试连接</button>
                            <button onClick={runDeploy}>推送并重启</button>
                        </div>
                    </div>
                    <div className="card">
                        <h3>推送日志</h3>
                        <pre>{deployLogs.join('\n')}</pre>
                    </div>
                </section>
            )}

            {tab === 'history' && (
                <section className="panel card">
                    <div className="row between">
                        <h3>快照历史</h3>
                        <button onClick={reloadHistory}>刷新</button>
                    </div>
                    <ul className="list">
                        {snapshots.map((snap) => (
                            <li key={snap}>
                                <span>{snap}</span>
                                <div className="row gap-8">
                                    <button onClick={() => restore(snap, 'env')}>恢复为 .env</button>
                                    <button onClick={() => restore(snap, 'agent')}>恢复为 agent_config.yaml</button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </section>
            )}
        </div>
    );
}
