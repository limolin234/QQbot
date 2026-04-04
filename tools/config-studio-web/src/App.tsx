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
import type {
    AgentConfigRoot,
    JsonObject,
    JsonValue,
    ScheduleConfig,
    StepNode,
    TabKey,
    TimelineEvent,
} from './types';

function newId() {
    return Math.random().toString(36).slice(2, 10);
}

function isObject(value: JsonValue | undefined): value is JsonObject {
    return !!value && typeof value === 'object' && !Array.isArray(value);
}

function asObject(value: JsonValue | undefined): JsonObject {
    return isObject(value) ? value : {};
}

function cloneValue<T>(value: T): T {
    return JSON.parse(JSON.stringify(value)) as T;
}

function parseScalarByType(raw: string, type: 'string' | 'number' | 'boolean') {
    if (type === 'number') {
        const num = Number(raw);
        return Number.isNaN(num) ? 0 : num;
    }
    if (type === 'boolean') {
        return raw === 'true';
    }
    return raw;
}

function getValueType(value: JsonValue): 'string' | 'number' | 'boolean' | 'null' | 'array' | 'object' {
    if (Array.isArray(value)) return 'array';
    if (value === null) return 'null';
    if (typeof value === 'object') return 'object';
    if (typeof value === 'number') return 'number';
    if (typeof value === 'boolean') return 'boolean';
    return 'string';
}

function createDefaultByKind(kind: 'string' | 'number' | 'boolean' | 'object' | 'array') {
    if (kind === 'number') return 0;
    if (kind === 'boolean') return false;
    if (kind === 'object') return {};
    if (kind === 'array') return [];
    return '';
}

type ParamField =
    | { key: string; label: string; kind: 'text' }
    | { key: string; label: string; kind: 'number' }
    | { key: string; label: string; kind: 'boolean' }
    | { key: string; label: string; kind: 'string_list' }
    | { key: string; label: string; kind: 'select'; options: string[] };

const ACTION_PARAM_SCHEMAS: Record<string, ParamField[]> = {
    'core.send_group_msg': [
        { key: 'group_id', label: '群号', kind: 'text' },
        { key: 'message', label: '消息内容', kind: 'text' },
    ],
    'core.send_private_msg': [
        { key: 'user_id', label: '用户QQ', kind: 'text' },
        { key: 'message', label: '消息内容', kind: 'text' },
    ],
    'dida.push_task_list': [
        { key: 'group_id', label: '群号', kind: 'text' },
        { key: 'user_qq', label: '用户QQ', kind: 'text' },
        {
            key: 'day_range',
            label: '时间范围',
            kind: 'select',
            options: ['today', 'tomorrow', 'week', 'all'],
        },
        { key: 'limit', label: '任务数量上限', kind: 'number' },
    ],
    'dida.poll': [
        { key: 'project_ids', label: '项目ID列表', kind: 'string_list' },
        { key: 'max_tasks_scan_per_user', label: '每用户扫描上限', kind: 'number' },
    ],
    'summary.daily_report': [
        {
            key: 'run_mode',
            label: '执行模式',
            kind: 'select',
            options: ['group', 'global'],
        },
        { key: 'send_to_groups', label: '目标群列表', kind: 'string_list' },
    ],
};

function ensureScheduler(agent: AgentConfigRoot): { timezone: string; schedules: ScheduleConfig[] } {
    const root = asObject(agent.scheduler_manager);
    if (!root.file_name) {
        root.file_name = 'scheduler_manager.py';
    }
    const config = asObject(root.config);
    if (!config.timezone || typeof config.timezone !== 'string') {
        config.timezone = 'Asia/Shanghai';
    }
    if (!Array.isArray(config.schedules)) {
        config.schedules = [];
    }
    root.config = config;
    agent.scheduler_manager = root;

    return {
        timezone: String(config.timezone),
        schedules: config.schedules as unknown as ScheduleConfig[],
    };
}

function normalizeStepsTree(nodes?: StepNode[]): StepNode[] {
    if (!nodes) return [];
    return nodes.map((node) => ({
        id: node.id || newId(),
        kind: node.kind,
        action: node.action || '',
        params: asObject(node.params),
        name: node.name || '',
        condition: node.condition || { source: 'env', key: '', op: 'eq', value: '' },
        children: normalizeStepsTree(node.children),
        then_steps: normalizeStepsTree(node.then_steps),
        else_steps: normalizeStepsTree(node.else_steps),
    }));
}

function StringListEditor({
    value,
    onChange,
}: {
    value: string[];
    onChange: (next: string[]) => void;
}) {
    return (
        <div className="string-list">
            {value.map((item, idx) => (
                <div key={`${item}_${idx}`} className="row gap-8">
                    <input
                        value={item}
                        onChange={(e) => {
                            const next = [...value];
                            next[idx] = e.target.value;
                            onChange(next);
                        }}
                    />
                    <button
                        onClick={() => {
                            const next = [...value];
                            next.splice(idx, 1);
                            onChange(next);
                        }}
                    >
                        删除
                    </button>
                </div>
            ))}
            <button onClick={() => onChange([...value, ''])}>+ 新增</button>
        </div>
    );
}

function ActionParamsEditor({
    action,
    params,
    onChange,
}: {
    action: string;
    params: JsonObject;
    onChange: (params: JsonObject) => void;
}) {
    const fields = ACTION_PARAM_SCHEMAS[action];
    if (!fields) {
        return (
            <div>
                <p>自定义动作参数</p>
                <JsonFormEditor value={params} onChange={(next) => onChange(asObject(next))} depth={0} />
            </div>
        );
    }

    return (
        <div className="grid2">
            {fields.map((field) => {
                const current = params[field.key] as JsonValue;
                if (field.kind === 'text') {
                    return (
                        <div key={field.key}>
                            <label>{field.label}</label>
                            <input
                                value={typeof current === 'string' ? current : ''}
                                onChange={(e) => onChange({ ...params, [field.key]: e.target.value })}
                            />
                        </div>
                    );
                }
                if (field.kind === 'number') {
                    return (
                        <div key={field.key}>
                            <label>{field.label}</label>
                            <input
                                type="number"
                                value={typeof current === 'number' ? current : 0}
                                onChange={(e) => onChange({ ...params, [field.key]: Number(e.target.value) })}
                            />
                        </div>
                    );
                }
                if (field.kind === 'boolean') {
                    return (
                        <label key={field.key} className="row gap-8">
                            <input
                                type="checkbox"
                                checked={Boolean(current)}
                                onChange={(e) => onChange({ ...params, [field.key]: e.target.checked })}
                            />
                            {field.label}
                        </label>
                    );
                }
                if (field.kind === 'string_list') {
                    return (
                        <div key={field.key} className="full-row">
                            <label>{field.label}</label>
                            <StringListEditor
                                value={Array.isArray(current) ? current.map(String) : []}
                                onChange={(next) => onChange({ ...params, [field.key]: next })}
                            />
                        </div>
                    );
                }
                return (
                    <div key={field.key}>
                        <label>{field.label}</label>
                        <select
                            value={typeof current === 'string' ? current : field.options[0]}
                            onChange={(e) => onChange({ ...params, [field.key]: e.target.value })}
                        >
                            {field.options.map((option) => (
                                <option key={option} value={option}>
                                    {option}
                                </option>
                            ))}
                        </select>
                    </div>
                );
            })}
        </div>
    );
}

function JsonFormEditor({
    value,
    onChange,
    depth,
}: {
    value: JsonValue;
    onChange: (next: JsonValue) => void;
    depth: number;
}) {
    const valueType = getValueType(value);

    if (valueType === 'string') {
        return (
            <input
                value={String(value)}
                onChange={(e) => onChange(e.target.value)}
                className={depth > 0 ? 'compact' : ''}
            />
        );
    }

    if (valueType === 'number') {
        return (
            <input
                type="number"
                value={Number(value)}
                onChange={(e) => onChange(Number(e.target.value))}
                className={depth > 0 ? 'compact' : ''}
            />
        );
    }

    if (valueType === 'boolean') {
        return (
            <label className="row gap-8">
                <input
                    type="checkbox"
                    checked={Boolean(value)}
                    onChange={(e) => onChange(e.target.checked)}
                />
                <span>{Boolean(value) ? 'true' : 'false'}</span>
            </label>
        );
    }

    if (valueType === 'null') {
        return <span className="muted">null（请改类型）</span>;
    }

    if (valueType === 'array') {
        const arr = value as JsonValue[];
        return (
            <div className="array-box">
                {arr.map((item, idx) => (
                    <div key={idx} className="array-item">
                        <JsonFormEditor
                            value={item}
                            depth={depth + 1}
                            onChange={(next) => {
                                const updated = [...arr];
                                updated[idx] = next;
                                onChange(updated);
                            }}
                        />
                        <button
                            onClick={() => {
                                const updated = [...arr];
                                updated.splice(idx, 1);
                                onChange(updated);
                            }}
                        >
                            删除
                        </button>
                    </div>
                ))}
                <div className="row gap-8 wrap">
                    {(['string', 'number', 'boolean', 'object', 'array'] as const).map((kind) => (
                        <button key={kind} onClick={() => onChange([...arr, createDefaultByKind(kind)])}>
                            + {kind}
                        </button>
                    ))}
                </div>
            </div>
        );
    }

    const obj = value as JsonObject;
    return (
        <div className="object-box">
            {Object.entries(obj).map(([k, v]) => (
                <div key={k} className="object-item">
                    <label>{k}</label>
                    <div className="row gap-8 align-start">
                        <div className="grow">
                            <JsonFormEditor
                                value={v}
                                depth={depth + 1}
                                onChange={(next) => onChange({ ...obj, [k]: next })}
                            />
                        </div>
                        <button
                            onClick={() => {
                                const next: JsonObject = { ...obj };
                                delete next[k];
                                onChange(next);
                            }}
                        >
                            删除键
                        </button>
                    </div>
                </div>
            ))}
            <ObjectAddField
                onAdd={(name, kind) => {
                    if (!name.trim()) return;
                    if (name in obj) return;
                    onChange({ ...obj, [name]: createDefaultByKind(kind) });
                }}
            />
        </div>
    );
}

function ObjectAddField({
    onAdd,
}: {
    onAdd: (name: string, kind: 'string' | 'number' | 'boolean' | 'object' | 'array') => void;
}) {
    const [name, setName] = useState('');
    const [kind, setKind] = useState<'string' | 'number' | 'boolean' | 'object' | 'array'>('string');

    return (
        <div className="row gap-8 wrap">
            <input placeholder="新字段名" value={name} onChange={(e) => setName(e.target.value)} />
            <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
                <option value="string">string</option>
                <option value="number">number</option>
                <option value="boolean">boolean</option>
                <option value="object">object</option>
                <option value="array">array</option>
            </select>
            <button
                onClick={() => {
                    onAdd(name, kind);
                    setName('');
                }}
            >
                + 添加字段
            </button>
        </div>
    );
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
                action: 'core.send_group_msg',
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
                            <label>动作</label>
                            <select
                                value={node.action || 'custom'}
                                onChange={(e) => updateNode(idx, { action: e.target.value })}
                            >
                                {Object.keys(ACTION_PARAM_SCHEMAS).map((actionKey) => (
                                    <option key={actionKey} value={actionKey}>
                                        {actionKey}
                                    </option>
                                ))}
                                <option value="custom">custom</option>
                            </select>
                            {(node.action || '') === 'custom' ? (
                                <input
                                    placeholder="输入完整 action id"
                                    value={node.action || ''}
                                    onChange={(e) => updateNode(idx, { action: e.target.value })}
                                />
                            ) : null}
                            <ActionParamsEditor
                                action={node.action || ''}
                                params={asObject(node.params)}
                                onChange={(params) => updateNode(idx, { params })}
                            />
                        </>
                    )}
                    {node.kind === 'group' && (
                        <>
                            <label>组名称</label>
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
                                <div>
                                    <label>source</label>
                                    <select
                                        value={node.condition?.source || 'env'}
                                        onChange={(e) =>
                                            updateNode(idx, {
                                                condition: { ...(node.condition || {}), source: e.target.value },
                                            })
                                        }
                                    >
                                        <option value="env">env</option>
                                        <option value="context">context</option>
                                    </select>
                                </div>
                                <div>
                                    <label>key</label>
                                    <input
                                        placeholder="key"
                                        value={node.condition?.key || ''}
                                        onChange={(e) =>
                                            updateNode(idx, {
                                                condition: { ...(node.condition || {}), key: e.target.value },
                                            })
                                        }
                                    />
                                </div>
                                <div>
                                    <label>op</label>
                                    <select
                                        value={node.condition?.op || 'eq'}
                                        onChange={(e) =>
                                            updateNode(idx, {
                                                condition: { ...(node.condition || {}), op: e.target.value },
                                            })
                                        }
                                    >
                                        <option value="eq">eq</option>
                                        <option value="ne">ne</option>
                                        <option value="contains">contains</option>
                                        <option value="in">in</option>
                                        <option value="truthy">truthy</option>
                                        <option value="falsy">falsy</option>
                                    </select>
                                </div>
                                <div>
                                    <label>value</label>
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
                setAgent((all.agent || {}) as AgentConfigRoot);
                setRawYaml(raw.raw_yaml || '');
                setLlmApiKey(all.env?.LLM_API_KEY || '');
                setLlmApiBaseUrl(all.env?.LLM_API_BASE_URL || '');

                const sections = Object.keys(all.agent || {});
                if (sections.length > 0) {
                    setActiveSection(sections[0]);
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
        setAgent(cloneValue(next));
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
            setAgent((all.agent || {}) as AgentConfigRoot);
            setStatus('saved');
        } catch (e) {
            setError(String(e));
        }
    };

    const addSection = () => {
        const base = 'new_section';
        let index = 1;
        let name = `${base}_${index}`;
        while (Object.keys(agent).includes(name)) {
            index += 1;
            name = `${base}_${index}`;
        }
        const next: AgentConfigRoot = {
            ...agent,
            [name]: {
                file_name: `${name}.py`,
                config: {},
            },
        };
        updateAgent(next);
        setActiveSection(name);
    };

    const removeSection = (name: string) => {
        const next: AgentConfigRoot = cloneValue(agent);
        delete next[name];
        updateAgent(next);
        const sections = Object.keys(next);
        setActiveSection(sections[0] || '');
    };

    const updateSectionValue = (name: string, value: JsonValue) => {
        const next: AgentConfigRoot = cloneValue(agent);
        next[name] = value;
        updateAgent(next);
    };

    const updateSchedules = (nextSchedules: ScheduleConfig[]) => {
        const next = cloneValue(agent as AgentConfigRoot);
        const schedulerManager = asObject(next.scheduler_manager);
        const config = asObject(schedulerManager.config);
        config.timezone = scheduler.timezone;
        config.schedules = nextSchedules as unknown as JsonValue;
        schedulerManager.file_name = (schedulerManager.file_name as string) || 'scheduler_manager.py';
        schedulerManager.config = config;
        next.scheduler_manager = schedulerManager;
        updateAgent(next);
    };

    const updateTimezone = (timezone: string) => {
        const next = cloneValue(agent as AgentConfigRoot);
        const schedulerManager = asObject(next.scheduler_manager);
        const config = asObject(schedulerManager.config);
        config.timezone = timezone;
        config.schedules = schedules as unknown as JsonValue;
        schedulerManager.file_name = (schedulerManager.file_name as string) || 'scheduler_manager.py';
        schedulerManager.config = config;
        next.scheduler_manager = schedulerManager;
        updateAgent(next);
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

    const sectionValue = activeSection ? (agent[activeSection] as JsonValue) : {};

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
                        <div className="row between">
                            <h3>Section 列表</h3>
                            <button onClick={addSection}>+ 新增 Section</button>
                        </div>
                        <ul className="list">
                            {Object.keys(agent).map((sectionName) => (
                                <li key={sectionName} className={activeSection === sectionName ? 'selected' : ''}>
                                    <button onClick={() => setActiveSection(sectionName)}>{sectionName}</button>
                                    {sectionName !== 'scheduler_manager' ? (
                                        <button onClick={() => removeSection(sectionName)}>删除</button>
                                    ) : null}
                                </li>
                            ))}
                        </ul>
                        <div className="row gap-8">
                            <button onClick={saveAgentFull}>保存全部配置</button>
                        </div>
                    </div>
                    <div className="card">
                        <h3>Section 表单编辑</h3>
                        {activeSection ? (
                            <>
                                <p>当前: {activeSection}</p>
                                <JsonFormEditor
                                    value={sectionValue}
                                    depth={0}
                                    onChange={(next) => updateSectionValue(activeSection, next)}
                                />
                                <div className="row gap-8 top-gap">
                                    <button onClick={saveAgentFull}>保存当前修改</button>
                                </div>
                            </>
                        ) : (
                            <p>暂无 Section，请先新增。</p>
                        )}
                    </div>
                    <div className="card full-row">
                        <details>
                            <summary>高级模式（原始 YAML）</summary>
                            <textarea rows={18} value={rawYaml} onChange={(e) => setRawYaml(e.target.value)} />
                            <button onClick={saveRaw}>保存原始 YAML</button>
                        </details>
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
                                    <button
                                        onClick={() => {
                                            const next = [...schedules];
                                            next.splice(idx, 1);
                                            updateSchedules(next);
                                            setSelectedSchedule(Math.max(0, idx - 1));
                                        }}
                                    >
                                        删除
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
                                <label>Enabled</label>
                                <label className="row gap-8">
                                    <input
                                        type="checkbox"
                                        checked={currentSchedule.enabled}
                                        onChange={(e) => {
                                            const next = [...schedules];
                                            next[selectedSchedule] = { ...currentSchedule, enabled: e.target.checked };
                                            updateSchedules(next);
                                        }}
                                    />
                                    {currentSchedule.enabled ? '已启用' : '已停用'}
                                </label>
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
                                        <div key={`${event.schedule_name}_${idx}`}>
                                            {event.trigger_at} | {event.schedule_name} | {event.source}
                                        </div>
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
                            if (k === 'auth_type') {
                                return (
                                    <div key={k}>
                                        <label>{k}</label>
                                        <select
                                            value={String(v)}
                                            onChange={(e) => setDeployForm({ ...deployForm, [k]: e.target.value })}
                                        >
                                            <option value="password">password</option>
                                            <option value="key">key</option>
                                        </select>
                                    </div>
                                );
                            }
                            if (k === 'restart_policy') {
                                return (
                                    <div key={k}>
                                        <label>{k}</label>
                                        <select
                                            value={String(v)}
                                            onChange={(e) => setDeployForm({ ...deployForm, [k]: e.target.value })}
                                        >
                                            <option value="docker-compose">docker-compose</option>
                                            <option value="systemctl">systemctl</option>
                                            <option value="pm2">pm2</option>
                                        </select>
                                    </div>
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
