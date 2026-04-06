import { useEffect, useMemo, useState } from 'react';
import {
    compileSchedules,
    deployPull,
    deployPush,
    getAllConfig,
    getTimeline,
    listSnapshots,
    restoreSnapshot,
    saveAgent,
    saveEnv,
    testConnection
} from './api';
import { TerminalPanel } from './TerminalPanel';
import type {
    AgentConfigRoot,
    JsonObject,
    JsonValue,
    ScheduleConfig,
    StepNode,
    TabKey,
    TimelineEvent
} from './types';

type FixedAgentKey = 'summary_config' | 'forward_config' | 'auto_reply_config' | 'dida_agent_config' | 'dida_config';

type AutoReplyRule = {
    enabled: boolean;
    chat_type: 'group' | 'private';
    number: string;
    trigger_mode: string;
    keywords: string[];
    ai_decision_prompt: string;
    reply_prompt: string;
    temperature?: number;
};

type DidaRule = {
    enabled: boolean;
    chat_type: 'group' | 'private';
    number: string;
    trigger_mode: string;
    dida_enabled: boolean;
    ai_decision_prompt: string;
    action_prompt: string;
    reply_prompt: string;
    action_temperature: number;
    reply_temperature: number;
};

type SummaryRule = {
    enabled: boolean;
    target_type: 'private' | 'group';
    target_id: string;
    run_mode: 'all' | 'auto' | 'manual';
};

type SummaryConfig = {
    model: string;
    temperature: number;
    max_line_chars: number;
    max_lines: number;
    summary_chat_scope: 'group' | 'private' | 'all';
    summary_group_filter_mode: 'all' | 'include' | 'exclude';
    summary_group_ids: string[];
    summary_global_overview: boolean;
    summary_send_mode: 'single_message' | 'multi_message';
    summary_group_reduce_enabled: boolean;
    rules: SummaryRule[];
};

type ForwardConfig = {
    model: string;
    decision_model?: string;
    temperature: number;
    monitor_group_qq_number: string[];
    forward_decision_prompt: string;
};

type AutoReplyConfig = {
    model: string;
    decision_model?: string;
    temperature: number;
    context_history_limit: number;
    context_max_chars: number;
    context_window_seconds: number;
    min_reply_interval_seconds: number;
    flush_check_interval_seconds: number;
    pending_expire_seconds: number;
    bypass_cooldown_when_at_bot: boolean;
    pending_max_messages: number;
    rules: AutoReplyRule[];
};

type DidaConfig = {
    client_id: string;
    client_secret: string;
    redirect_uri: string;
    model: string;
    decision_model?: string;
    temperature: number;
    bot_qq: string;
    admin_qqs: string[];
    due_window_seconds: number;
    compensate_window_seconds: number;
    max_tasks_scan_per_user: number;
    reminder_group_ids: string[];
    project_ids: string[];
    context_history_limit: number;
    context_max_chars: number;
    context_window_seconds: number;
    min_reply_interval_seconds: number;
    flush_check_interval_seconds: number;
    pending_expire_seconds: number;
    bypass_cooldown_when_at_bot: boolean;
    pending_max_messages: number;
    rules: DidaRule[];
};

type SchedulerManagerConfig = {
    timezone: string;
    schedules: ScheduleConfig[];
};

type ActionFieldType = 'string' | 'number' | 'boolean' | 'string[]';

type ActionFieldSchema = {
    key: string;
    label: string;
    type: ActionFieldType;
    options?: ActionFieldOption[];
    defaultValue?: JsonValue;
};

type ActionFieldOption = { value: string; label: string };

type ActionSchema = {
    action: string;
    label: string;
    fields: ActionFieldSchema[];
};

// Centralized enum options for action params. Future enum params should reuse these.
const ACTION_PARAM_ENUMS = {
    summaryRunMode: [
        { value: 'auto', label: 'auto（当天汇总）' },
        { value: 'manual', label: 'manual（增量汇总）' },
    ] as ActionFieldOption[],
    summaryTargetType: [
        { value: 'private', label: 'private（私聊）' },
        { value: 'group', label: 'group（群聊）' },
    ] as ActionFieldOption[],
    didaDayRange: [
        { value: 'today', label: 'today（今天+已过期）' },
        { value: 'overdue', label: 'overdue（仅已过期）' },
        { value: 'tomorrow', label: 'tomorrow（明天）' },
        { value: 'all', label: 'all（全部）' },
    ] as ActionFieldOption[],
};

function stringField(key: string, label: string, defaultValue = ''): ActionFieldSchema {
    return { key, label, type: 'string', defaultValue };
}

function numberField(key: string, label: string, defaultValue = 0): ActionFieldSchema {
    return { key, label, type: 'number', defaultValue };
}

function stringArrayField(key: string, label: string, defaultValue: string[] = []): ActionFieldSchema {
    return { key, label, type: 'string[]', defaultValue };
}

function enumField(
    key: string,
    label: string,
    options: ActionFieldOption[],
    defaultValue: string,
): ActionFieldSchema {
    return {
        key,
        label,
        type: 'string',
        options,
        defaultValue,
    };
}

const ACTION_SCHEMAS: ActionSchema[] = [
    {
        action: 'core.send_group_msg',
        label: '发送群消息',
        fields: [stringField('group_id', '群号'), stringField('message', '消息内容')],
    },
    {
        action: 'summary.daily_report',
        label: '每日总结',
        fields: [
            enumField('run_mode', '运行模式', ACTION_PARAM_ENUMS.summaryRunMode, 'auto'),
        ],
    },
    {
        action: 'dida.poll',
        label: '轮询任务',
        fields: [],
    },
    {
        action: 'dida.push_task_list',
        label: '推送任务清单',
        fields: [
            stringField('group_id', '群号'),
            stringField('user_qq', '用户 QQ'),
            enumField('day_range', '时间范围', ACTION_PARAM_ENUMS.didaDayRange, 'today'),
            numberField('limit', '条数上限', 20),
        ],
    },
];

const ACTION_OPTIONS = ACTION_SCHEMAS.map((item) => ({ value: item.action, label: `${item.action} (${item.label})` }));

const CHAT_TYPE_LABELS: Record<'group' | 'private', string> = {
    group: '群聊',
    private: '私聊',
};

const AUTO_TRIGGER_MODE_LABELS: Record<string, string> = {
    always: '始终触发',
    keyword: '关键词触发',
    at_bot: '@机器人触发',
    ai_decide: 'AI 决策触发',
    'ai_decide || keyword': 'AI 决策或关键词触发',
    'at_bot || keyword': '@机器人或关键词触发',
    'at_bot || ai_decide': '@机器人或 AI 决策触发',
    'at_bot || ai_decide || keyword': '@机器人、AI 决策或关键词触发',
};

const DIDA_TRIGGER_MODE_LABELS: Record<string, string> = {
    ai_decide: 'AI 决策触发',
    always: '始终触发',
    keyword: '关键词触发',
    at_bot: '@机器人触发',
    'ai_decide || keyword': 'AI 决策或关键词触发',
    'at_bot || keyword': '@机器人或关键词触发',
    'at_bot || ai_decide': '@机器人或 AI 决策触发',
    'at_bot || ai_decide || keyword': '@机器人、AI 决策或关键词触发',
};

type FixedSection = {
    file_name: string;
    config: JsonObject;
};

const FIXED_AGENT_OPTIONS: Array<{ key: FixedAgentKey; label: string }> = [
    { key: 'summary_config', label: '总结 Agent' },
    { key: 'forward_config', label: '转发 Agent' },
    { key: 'auto_reply_config', label: '自动回复 Agent' },
    { key: 'dida_agent_config', label: '滴答 Agent' },
];

function asObject(value: JsonValue | undefined): JsonObject {
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as JsonObject) : {};
}

function toStringArray(value: JsonValue | undefined): string[] {
    if (!Array.isArray(value)) return [];
    return value.map((item) => String(item));
}

function toNumber(value: JsonValue | undefined, fallback: number): number {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function toBool(value: JsonValue | undefined, fallback: boolean): boolean {
    if (typeof value === 'boolean') return value;
    return fallback;
}

function toStringValue(value: JsonValue | undefined, fallback = ''): string {
    if (typeof value === 'string') return value;
    if (value === undefined || value === null) return fallback;
    return String(value);
}

function buildDidaAgentPersistedConfig(source: JsonObject): JsonObject {
    return {
        model: toStringValue(source.model),
        decision_model: toStringValue(source.decision_model),
        temperature: toNumber(source.temperature, 0.1),
        bot_qq: toStringValue(source.bot_qq),
        admin_qqs: toStringArray(source.admin_qqs),
        context_history_limit: toNumber(source.context_history_limit, 50),
        context_max_chars: toNumber(source.context_max_chars, 2000),
        context_window_seconds: toNumber(source.context_window_seconds, 0),
        min_reply_interval_seconds: toNumber(source.min_reply_interval_seconds, 10),
        flush_check_interval_seconds: toNumber(source.flush_check_interval_seconds, 10),
        pending_expire_seconds: toNumber(source.pending_expire_seconds, 3600),
        bypass_cooldown_when_at_bot: toBool(source.bypass_cooldown_when_at_bot, true),
        pending_max_messages: toNumber(source.pending_max_messages, 50),
        rules: Array.isArray(source.rules) ? source.rules : [makeDefaultDidaRule() as unknown as JsonValue],
    };
}

function makeNodeId(): string {
    return `node_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function getActionSchema(action: string): ActionSchema | undefined {
    return ACTION_SCHEMAS.find((item) => item.action === action);
}

function withActionDefaults(action: string, params: JsonObject, pruneUnknown = false): JsonObject {
    const schema = getActionSchema(action);
    if (!schema) return params;

    const next: JsonObject = {};
    const allowedKeys = new Set(schema.fields.map((field) => field.key));

    if (pruneUnknown) {
        for (const key of Object.keys(params)) {
            if (allowedKeys.has(key)) {
                next[key] = params[key];
            }
        }
    } else {
        Object.assign(next, params);
    }

    for (const field of schema.fields) {
        if (next[field.key] === undefined && field.defaultValue !== undefined) {
            next[field.key] = field.defaultValue;
        }
    }
    return next;
}

function toStepNode(item: JsonValue): StepNode | null {
    const obj = asObject(item);
    const kind = toStringValue(obj.kind, 'action') as StepNode['kind'];
    if (kind !== 'action' && kind !== 'group') {
        return null;
    }

    const node: StepNode = {
        id: toStringValue(obj.id, makeNodeId()),
        kind,
    };

    if (kind === 'action') {
        node.action = toStringValue(obj.action, 'core.send_group_msg');
        node.params = withActionDefaults(node.action, asObject(obj.params), true);
        return node;
    }

    if (kind === 'group') {
        node.name = toStringValue(obj.name, 'group');
        const children = Array.isArray(obj.children) ? obj.children : [];
        node.children = children.map((child) => toStepNode(child)).filter((child): child is StepNode => Boolean(child));
        return node;
    }

    return null;
}

function normalizeSchedules(input: JsonValue | undefined): ScheduleConfig[] {
    if (!Array.isArray(input)) return [];
    const items: ScheduleConfig[] = [];
    input.forEach((raw, idx) => {
        const obj = asObject(raw);
        const scheduleType = toStringValue(obj.type, 'cron') as 'cron' | 'interval';
        const schedule: ScheduleConfig = {
            name: toStringValue(obj.name, `schedule_${idx + 1}`),
            type: scheduleType,
            enabled: toBool(obj.enabled, true),
        };
        if (scheduleType === 'cron') {
            schedule.expression = toStringValue(obj.expression, '*/5 * * * *');
        } else {
            schedule.seconds = toNumber(obj.seconds, 60);
        }

        if (Array.isArray(obj.steps_tree)) {
            schedule.steps_tree = obj.steps_tree
                .map((item) => toStepNode(item))
                .filter((item): item is StepNode => Boolean(item));
        } else if (Array.isArray(obj.steps)) {
            schedule.steps_tree = obj.steps.map((legacy) => {
                const legacyObj = asObject(legacy);
                const action = toStringValue(legacyObj.action, 'core.send_group_msg');
                return {
                    id: makeNodeId(),
                    kind: 'action',
                    action,
                    params: withActionDefaults(action, asObject(legacyObj.params), true),
                } satisfies StepNode;
            });
        } else {
            schedule.steps_tree = [];
        }

        items.push(schedule);
    });
    return items;
}

function getSchedulerManagerConfig(agent: AgentConfigRoot): SchedulerManagerConfig {
    const section = asObject(agent.scheduler_manager);
    const config = asObject(section.config);
    return {
        timezone: toStringValue(config.timezone, 'Asia/Shanghai'),
        schedules: normalizeSchedules(config.schedules),
    };
}

function isSchedulerManagerEnabled(agent: AgentConfigRoot): boolean {
    const section = asObject(agent.scheduler_manager);
    const config = asObject(section.config);
    return Object.keys(config).length > 0;
}

function hasActionInStepNodes(nodes: StepNode[] | undefined, actionName: string): boolean {
    if (!nodes || nodes.length === 0) return false;
    return nodes.some((node) => {
        if (node.kind === 'action') {
            return (node.action || '').trim() === actionName;
        }
        return hasActionInStepNodes(node.children || [], actionName);
    });
}

function defaultSchedule(): ScheduleConfig {
    return {
        name: 'new_schedule',
        type: 'cron',
        expression: '*/5 * * * *',
        enabled: true,
        steps_tree: [],
    };
}

function defaultStep(kind: StepNode['kind']): StepNode {
    if (kind === 'group') {
        return { id: makeNodeId(), kind: 'group', name: 'group', children: [] };
    }
    const action = 'core.send_group_msg';
    return {
        id: makeNodeId(),
        kind: 'action',
        action,
        params: withActionDefaults(action, {}, true),
    };
}

function makeDefaultAutoRule(): AutoReplyRule {
    return {
        enabled: true,
        chat_type: 'group',
        number: '',
        trigger_mode: 'always',
        keywords: [],
        ai_decision_prompt: '',
        reply_prompt: '',
        temperature: 0.4,
    };
}

function makeDefaultDidaRule(): DidaRule {
    return {
        enabled: true,
        chat_type: 'group',
        number: '',
        trigger_mode: 'ai_decide',
        dida_enabled: true,
        ai_decision_prompt: '',
        action_prompt: '',
        reply_prompt: '',
        action_temperature: 0,
        reply_temperature: 0.5,
    };
}

function makeDefaultSummaryRule(): SummaryRule {
    return {
        enabled: true,
        target_type: 'private',
        target_id: '',
        run_mode: 'all',
    };
}

function defaultSections(): Record<FixedAgentKey, FixedSection> {
    return {
        summary_config: {
            file_name: 'summary.py',
            config: {
                model: '',
                temperature: 0.2,
                max_line_chars: 300,
                max_lines: 500,
                summary_chat_scope: 'group',
                summary_group_filter_mode: 'all',
                summary_group_ids: [],
                summary_global_overview: true,
                summary_send_mode: 'multi_message',
                summary_group_reduce_enabled: true,
                rules: [makeDefaultSummaryRule()],
            },
        },
        forward_config: {
            file_name: 'forward.py',
            config: {
                model: '',
                decision_model: '',
                temperature: 0,
                monitor_group_qq_number: [],
                forward_decision_prompt: '',
            },
        },
        auto_reply_config: {
            file_name: 'auto_reply.py',
            config: {
                model: '',
                decision_model: '',
                temperature: 0.4,
                context_history_limit: 50,
                context_max_chars: 2000,
                context_window_seconds: 0,
                min_reply_interval_seconds: 10,
                flush_check_interval_seconds: 10,
                pending_expire_seconds: 3600,
                bypass_cooldown_when_at_bot: false,
                pending_max_messages: 50,
                rules: [makeDefaultAutoRule()],
            },
        },
        dida_agent_config: {
            file_name: 'dida_agent.py',
            config: {
                model: '',
                decision_model: '',
                temperature: 0.1,
                bot_qq: '',
                admin_qqs: [],
                context_history_limit: 50,
                context_max_chars: 2000,
                context_window_seconds: 0,
                min_reply_interval_seconds: 10,
                flush_check_interval_seconds: 10,
                pending_expire_seconds: 3600,
                bypass_cooldown_when_at_bot: true,
                pending_max_messages: 50,
                rules: [makeDefaultDidaRule()],
            },
        },
        dida_config: {
            file_name: 'dida_scheduler.py',
            config: {
                client_id: '',
                client_secret: '',
                redirect_uri: '',
            },
        },
    };
}

function normalizeFixedSections(agent: AgentConfigRoot): Record<FixedAgentKey, FixedSection> {
    const defaults = defaultSections();
    const keys = Object.keys(defaults) as FixedAgentKey[];
    for (const key of keys) {
        let rawSection = asObject(agent[key]);
        if (key === 'dida_config') {
            const legacy = asObject(agent.dida_scheduler_config);
            if (Object.keys(rawSection).length === 0 && Object.keys(legacy).length > 0) {
                rawSection = legacy;
            }
        }
        const fileName = toStringValue(rawSection.file_name, defaults[key].file_name);
        const cfg = asObject(rawSection.config);
        defaults[key] = {
            file_name: fileName || defaults[key].file_name,
            config: { ...defaults[key].config, ...cfg },
        };
    }
    return defaults;
}

function cloneAgent(agent: AgentConfigRoot): AgentConfigRoot {
    return JSON.parse(JSON.stringify(agent)) as AgentConfigRoot;
}

function StringListEditor({
    values,
    onChange,
}: {
    values: string[];
    onChange: (next: string[]) => void;
}) {
    return (
        <div className="string-list">
            {values.map((item, idx) => (
                <div className="row gap-8" key={`${item}_${idx}`}>
                    <input
                        value={item}
                        onChange={(e) => {
                            const next = [...values];
                            next[idx] = e.target.value;
                            onChange(next);
                        }}
                    />
                    <button
                        onClick={() => {
                            const next = [...values];
                            next.splice(idx, 1);
                            onChange(next);
                        }}
                    >
                        删除
                    </button>
                </div>
            ))}
            <button onClick={() => onChange([...values, ''])}>+ 添加</button>
        </div>
    );
}

function SwitchField({
    checked,
    onChange,
    label,
}: {
    checked: boolean;
    onChange: (next: boolean) => void;
    label: string;
}) {
    return (
        <div className="switch-field">
            <span>{label}</span>
            <label className="switch" aria-label={label}>
                <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
                <span className="slider" />
            </label>
        </div>
    );
}

function FormModal({
    open,
    title,
    onClose,
    children,
}: {
    open: boolean;
    title: string;
    onClose: () => void;
    children: JSX.Element;
}) {
    if (!open) return null;
    return (
        <div className="modal-mask" onClick={onClose}>
            <div className="modal-card" onClick={(e) => e.stopPropagation()}>
                <div className="row between">
                    <h3>{title}</h3>
                    <button onClick={onClose}>关闭</button>
                </div>
                <div className="modal-body">{children}</div>
                <div className="row gap-8 top-gap">
                    <button onClick={onClose}>保存并返回</button>
                </div>
            </div>
        </div>
    );
}

function SummaryRulesEditor({
    rules,
    onChange,
}: {
    rules: SummaryRule[];
    onChange: (next: SummaryRule[]) => void;
}) {
    const [editingIdx, setEditingIdx] = useState<number | null>(null);

    const updateRule = (idx: number, patch: Partial<SummaryRule>) => {
        const next = [...rules];
        next[idx] = { ...next[idx], ...patch };
        onChange(next);
    };

    const editingRule = editingIdx !== null ? rules[editingIdx] : null;

    return (
        <div className="schedule-grid">
            {rules.map((rule, idx) => (
                <div key={idx} className="rule-card">
                    <div className="row between">
                        <strong>规则 {idx + 1}</strong>
                        <div className="row gap-8">
                            <SwitchField checked={rule.enabled} onChange={(enabled) => updateRule(idx, { enabled })} label="启用规则" />
                            <button onClick={() => setEditingIdx(idx)}>编辑规则</button>
                            <button
                                onClick={() => {
                                    const next = [...rules];
                                    next.splice(idx, 1);
                                    onChange(next);
                                }}
                            >
                                删除规则
                            </button>
                        </div>
                    </div>
                    <div className="rule-preview">
                        {rule.target_type} | {rule.target_id || '未填写目标号'} | mode={rule.run_mode}
                    </div>
                </div>
            ))}
            <button onClick={() => onChange([...rules, makeDefaultSummaryRule()])}>+ 添加规则</button>

            <FormModal
                open={editingRule !== null}
                title={editingIdx !== null ? `规则 ${editingIdx + 1} 配置` : '规则配置'}
                onClose={() => setEditingIdx(null)}
            >
                {editingRule ? (
                    <div className="grid2">
                        <SwitchField
                            checked={editingRule.enabled}
                            onChange={(enabled) => editingIdx !== null && updateRule(editingIdx, { enabled })}
                            label="启用规则"
                        />
                        <div>
                            <label>target_type</label>
                            <select
                                value={editingRule.target_type}
                                onChange={(e) =>
                                    editingIdx !== null && updateRule(editingIdx, { target_type: e.target.value as 'private' | 'group' })
                                }
                            >
                                <option value="private">private（私聊）</option>
                                <option value="group">group（群聊）</option>
                            </select>
                        </div>
                        <div>
                            <label>target_id（群号/QQ）</label>
                            <input
                                value={editingRule.target_id}
                                onChange={(e) => editingIdx !== null && updateRule(editingIdx, { target_id: e.target.value })}
                            />
                        </div>
                        <div>
                            <label>run_mode</label>
                            <select
                                value={editingRule.run_mode}
                                onChange={(e) =>
                                    editingIdx !== null && updateRule(editingIdx, { run_mode: e.target.value as 'all' | 'auto' | 'manual' })
                                }
                            >
                                <option value="all">all（全部）</option>
                                <option value="auto">auto（仅自动）</option>
                                <option value="manual">manual（仅手动）</option>
                            </select>
                        </div>
                    </div>
                ) : (
                    <div />
                )}
            </FormModal>
        </div>
    );
}

function SummaryForm({ value, onChange }: { value: SummaryConfig; onChange: (next: SummaryConfig) => void }) {
    const [openBasicModal, setOpenBasicModal] = useState(false);

    return (
        <div>
            <div className="summary-card">
                <div className="row between wrap gap-8">
                    <strong>基础参数</strong>
                    <button onClick={() => setOpenBasicModal(true)}>编辑基础参数</button>
                </div>
                <div className="rule-preview">
                    模型：{value.model || '未设置'} | 温度：{value.temperature} | send_mode：{value.summary_send_mode}
                </div>
                <div className="rule-preview">
                    scope：{value.summary_chat_scope} | filter：{value.summary_group_filter_mode} | 规则数：{value.rules.length}
                </div>
            </div>

            <FormModal open={openBasicModal} title="总结 Agent 基础参数" onClose={() => setOpenBasicModal(false)}>
                <div className="grid2">
                    <div>
                        <label>model</label>
                        <input value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })} />
                    </div>
                    <div>
                        <label>temperature</label>
                        <input
                            type="number"
                            step="0.1"
                            value={value.temperature}
                            onChange={(e) => onChange({ ...value, temperature: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>max_line_chars</label>
                        <input
                            type="number"
                            value={value.max_line_chars}
                            onChange={(e) => onChange({ ...value, max_line_chars: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>max_lines</label>
                        <input
                            type="number"
                            value={value.max_lines}
                            onChange={(e) => onChange({ ...value, max_lines: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>summary_chat_scope</label>
                        <select
                            value={value.summary_chat_scope}
                            onChange={(e) =>
                                onChange({ ...value, summary_chat_scope: e.target.value as SummaryConfig['summary_chat_scope'] })
                            }
                        >
                            <option value="group">group</option>
                            <option value="private">private</option>
                            <option value="all">all</option>
                        </select>
                    </div>
                    <div>
                        <label>summary_group_filter_mode</label>
                        <select
                            value={value.summary_group_filter_mode}
                            onChange={(e) =>
                                onChange({
                                    ...value,
                                    summary_group_filter_mode: e.target.value as SummaryConfig['summary_group_filter_mode'],
                                })
                            }
                        >
                            <option value="all">all</option>
                            <option value="include">include</option>
                            <option value="exclude">exclude</option>
                        </select>
                    </div>
                    <div className="full-row">
                        <label>summary_group_ids</label>
                        <StringListEditor
                            values={value.summary_group_ids}
                            onChange={(summary_group_ids) => onChange({ ...value, summary_group_ids })}
                        />
                    </div>
                    <SwitchField
                        checked={value.summary_global_overview}
                        onChange={(summary_global_overview) => onChange({ ...value, summary_global_overview })}
                        label="summary_global_overview"
                    />
                    <div>
                        <label>summary_send_mode</label>
                        <select
                            value={value.summary_send_mode}
                            onChange={(e) =>
                                onChange({ ...value, summary_send_mode: e.target.value as SummaryConfig['summary_send_mode'] })
                            }
                        >
                            <option value="single_message">single_message</option>
                            <option value="multi_message">multi_message</option>
                        </select>
                    </div>
                    <SwitchField
                        checked={value.summary_group_reduce_enabled}
                        onChange={(summary_group_reduce_enabled) => onChange({ ...value, summary_group_reduce_enabled })}
                        label="summary_group_reduce_enabled"
                    />
                </div>
            </FormModal>

            <div className="top-gap">
                <label>规则列表</label>
                <SummaryRulesEditor rules={value.rules} onChange={(rules) => onChange({ ...value, rules })} />
            </div>
        </div>
    );
}

function ForwardForm({ value, onChange }: { value: ForwardConfig; onChange: (next: ForwardConfig) => void }) {
    return (
        <div className="grid2">
            <div>
                <label>model</label>
                <input value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })} />
            </div>
            <div>
                <label>decision_model</label>
                <input
                    value={value.decision_model || ''}
                    onChange={(e) => onChange({ ...value, decision_model: e.target.value })}
                />
            </div>
            <div>
                <label>temperature</label>
                <input
                    type="number"
                    step="0.1"
                    value={value.temperature}
                    onChange={(e) => onChange({ ...value, temperature: Number(e.target.value) })}
                />
            </div>
            <div className="full-row">
                <label>monitor_group_qq_number</label>
                <StringListEditor
                    values={value.monitor_group_qq_number}
                    onChange={(monitor_group_qq_number) => onChange({ ...value, monitor_group_qq_number })}
                />
            </div>
            <div className="full-row">
                <label>forward_decision_prompt</label>
                <textarea
                    rows={10}
                    value={value.forward_decision_prompt}
                    onChange={(e) => onChange({ ...value, forward_decision_prompt: e.target.value })}
                />
            </div>
        </div>
    );
}

function AutoReplyRulesEditor({
    rules,
    onChange,
}: {
    rules: AutoReplyRule[];
    onChange: (next: AutoReplyRule[]) => void;
}) {
    const [editingBasicIdx, setEditingBasicIdx] = useState<number | null>(null);
    const [editingPromptIdx, setEditingPromptIdx] = useState<number | null>(null);

    const updateRule = (idx: number, patch: Partial<AutoReplyRule>) => {
        const next = [...rules];
        next[idx] = { ...next[idx], ...patch };
        onChange(next);
    };

    const basicRule = editingBasicIdx !== null ? rules[editingBasicIdx] : null;
    const promptRule = editingPromptIdx !== null ? rules[editingPromptIdx] : null;

    return (
        <div className="schedule-grid">
            {rules.map((rule, idx) => (
                <div key={idx} className="rule-card">
                    <div className="row between">
                        <strong>规则 {idx + 1}</strong>
                        <div className="row gap-8">
                            <SwitchField checked={rule.enabled} onChange={(enabled) => updateRule(idx, { enabled })} label="启用规则" />
                            <button onClick={() => setEditingBasicIdx(idx)}>编辑基础参数</button>
                            <button onClick={() => setEditingPromptIdx(idx)}>编辑提示词</button>
                            <button
                                onClick={() => {
                                    const next = [...rules];
                                    next.splice(idx, 1);
                                    onChange(next);
                                }}
                            >
                                删除规则
                            </button>
                        </div>
                    </div>
                    <div className="rule-preview">
                        {CHAT_TYPE_LABELS[rule.chat_type]} | {AUTO_TRIGGER_MODE_LABELS[rule.trigger_mode] || rule.trigger_mode} | {rule.number || '未填写目标号'}
                    </div>
                </div>
            ))}
            <button onClick={() => onChange([...rules, makeDefaultAutoRule()])}>+ 添加规则</button>

            <FormModal
                open={basicRule !== null}
                title={editingBasicIdx !== null ? `规则 ${editingBasicIdx + 1} 基础参数` : '基础参数'}
                onClose={() => setEditingBasicIdx(null)}
            >
                {basicRule ? (
                    <div className="grid2">
                        <SwitchField
                            checked={basicRule.enabled}
                            onChange={(enabled) => editingBasicIdx !== null && updateRule(editingBasicIdx, { enabled })}
                            label="启用规则"
                        />
                        <div>
                            <label>聊天类型</label>
                            <select
                                value={basicRule.chat_type}
                                onChange={(e) =>
                                    editingBasicIdx !== null && updateRule(editingBasicIdx, { chat_type: e.target.value as 'group' | 'private' })
                                }
                            >
                                <option value="group">群聊</option>
                                <option value="private">私聊</option>
                            </select>
                        </div>
                        <div>
                            <label>目标号（群号/QQ）</label>
                            <input
                                value={basicRule.number}
                                onChange={(e) => editingBasicIdx !== null && updateRule(editingBasicIdx, { number: e.target.value })}
                            />
                        </div>
                        <div>
                            <label>触发模式</label>
                            <select
                                value={basicRule.trigger_mode}
                                onChange={(e) => editingBasicIdx !== null && updateRule(editingBasicIdx, { trigger_mode: e.target.value })}
                            >
                                <option value="always">始终触发</option>
                                <option value="keyword">关键词触发</option>
                                <option value="at_bot">@机器人触发</option>
                                <option value="ai_decide">AI 决策触发</option>
                                <option value="at_bot || ai_decide">@机器人或 AI 决策触发</option>
                                <option value="ai_decide || keyword">AI 决策或关键词触发</option>
                                <option value="at_bot || keyword">@机器人或关键词触发</option>
                            </select>
                        </div>
                        <div>
                            <label>温度（可选）</label>
                            <input
                                type="number"
                                step="0.1"
                                value={basicRule.temperature ?? 0.4}
                                onChange={(e) => editingBasicIdx !== null && updateRule(editingBasicIdx, { temperature: Number(e.target.value) })}
                            />
                        </div>
                        <div className="full-row">
                            <label>关键词列表</label>
                            <StringListEditor
                                values={basicRule.keywords}
                                onChange={(keywords) => editingBasicIdx !== null && updateRule(editingBasicIdx, { keywords })}
                            />
                        </div>
                    </div>
                ) : (
                    <div />
                )}
            </FormModal>

            <FormModal
                open={promptRule !== null}
                title={editingPromptIdx !== null ? `规则 ${editingPromptIdx + 1} 提示词` : '提示词'}
                onClose={() => setEditingPromptIdx(null)}
            >
                {promptRule ? (
                    <div>
                        <label>AI 决策提示词</label>
                        <textarea
                            rows={8}
                            value={promptRule.ai_decision_prompt}
                            onChange={(e) => editingPromptIdx !== null && updateRule(editingPromptIdx, { ai_decision_prompt: e.target.value })}
                        />
                        <label>回复提示词</label>
                        <textarea
                            rows={8}
                            value={promptRule.reply_prompt}
                            onChange={(e) => editingPromptIdx !== null && updateRule(editingPromptIdx, { reply_prompt: e.target.value })}
                        />
                    </div>
                ) : (
                    <div />
                )}
            </FormModal>
        </div>
    );
}

function DidaRulesEditor({ rules, onChange }: { rules: DidaRule[]; onChange: (next: DidaRule[]) => void }) {
    const [editingBasicIdx, setEditingBasicIdx] = useState<number | null>(null);
    const [editingPromptIdx, setEditingPromptIdx] = useState<number | null>(null);

    const updateRule = (idx: number, patch: Partial<DidaRule>) => {
        const next = [...rules];
        next[idx] = { ...next[idx], ...patch };
        onChange(next);
    };

    const basicRule = editingBasicIdx !== null ? rules[editingBasicIdx] : null;
    const promptRule = editingPromptIdx !== null ? rules[editingPromptIdx] : null;

    return (
        <div className="schedule-grid">
            {rules.map((rule, idx) => (
                <div key={idx} className="rule-card">
                    <div className="row between">
                        <strong>规则 {idx + 1}</strong>
                        <div className="row gap-8">
                            <SwitchField checked={rule.enabled} onChange={(enabled) => updateRule(idx, { enabled })} label="启用规则" />
                            <SwitchField
                                checked={rule.dida_enabled}
                                onChange={(dida_enabled) => updateRule(idx, { dida_enabled })}
                                label="启用滴答同步"
                            />
                            <button onClick={() => setEditingBasicIdx(idx)}>编辑基础参数</button>
                            <button onClick={() => setEditingPromptIdx(idx)}>编辑提示词</button>
                            <button
                                onClick={() => {
                                    const next = [...rules];
                                    next.splice(idx, 1);
                                    onChange(next);
                                }}
                            >
                                删除规则
                            </button>
                        </div>
                    </div>
                    <div className="rule-preview">
                        {CHAT_TYPE_LABELS[rule.chat_type]} | {DIDA_TRIGGER_MODE_LABELS[rule.trigger_mode] || rule.trigger_mode} | {rule.number || '未填写目标号'}
                    </div>
                </div>
            ))}
            <button onClick={() => onChange([...rules, makeDefaultDidaRule()])}>+ 添加规则</button>

            <FormModal
                open={basicRule !== null}
                title={editingBasicIdx !== null ? `规则 ${editingBasicIdx + 1} 基础参数` : '基础参数'}
                onClose={() => setEditingBasicIdx(null)}
            >
                {basicRule ? (
                    <div className="grid2">
                        <SwitchField
                            checked={basicRule.enabled}
                            onChange={(enabled) => editingBasicIdx !== null && updateRule(editingBasicIdx, { enabled })}
                            label="启用规则"
                        />
                        <SwitchField
                            checked={basicRule.dida_enabled}
                            onChange={(dida_enabled) => editingBasicIdx !== null && updateRule(editingBasicIdx, { dida_enabled })}
                            label="启用滴答同步"
                        />
                        <div>
                            <label>聊天类型</label>
                            <select
                                value={basicRule.chat_type}
                                onChange={(e) =>
                                    editingBasicIdx !== null && updateRule(editingBasicIdx, { chat_type: e.target.value as 'group' | 'private' })
                                }
                            >
                                <option value="group">群聊</option>
                                <option value="private">私聊</option>
                            </select>
                        </div>
                        <div>
                            <label>目标号（群号/QQ）</label>
                            <input
                                value={basicRule.number}
                                onChange={(e) => editingBasicIdx !== null && updateRule(editingBasicIdx, { number: e.target.value })}
                            />
                        </div>
                        <div>
                            <label>触发模式</label>
                            <select
                                value={basicRule.trigger_mode}
                                onChange={(e) => editingBasicIdx !== null && updateRule(editingBasicIdx, { trigger_mode: e.target.value })}
                            >
                                <option value="ai_decide">AI 决策触发</option>
                                <option value="always">始终触发</option>
                                <option value="keyword">关键词触发</option>
                                <option value="at_bot">@机器人触发</option>
                                <option value="at_bot || ai_decide">@机器人或 AI 决策触发</option>
                                <option value="ai_decide || keyword">AI 决策或关键词触发</option>
                                <option value="at_bot || keyword">@机器人或关键词触发</option>
                            </select>
                        </div>
                        <div>
                            <label>动作温度</label>
                            <input
                                type="number"
                                step="0.1"
                                value={basicRule.action_temperature}
                                onChange={(e) => editingBasicIdx !== null && updateRule(editingBasicIdx, { action_temperature: Number(e.target.value) })}
                            />
                        </div>
                        <div>
                            <label>回复温度</label>
                            <input
                                type="number"
                                step="0.1"
                                value={basicRule.reply_temperature}
                                onChange={(e) => editingBasicIdx !== null && updateRule(editingBasicIdx, { reply_temperature: Number(e.target.value) })}
                            />
                        </div>
                    </div>
                ) : (
                    <div />
                )}
            </FormModal>

            <FormModal
                open={promptRule !== null}
                title={editingPromptIdx !== null ? `规则 ${editingPromptIdx + 1} 提示词` : '提示词'}
                onClose={() => setEditingPromptIdx(null)}
            >
                {promptRule ? (
                    <div>
                        <label>AI 决策提示词</label>
                        <textarea
                            rows={7}
                            value={promptRule.ai_decision_prompt}
                            onChange={(e) => editingPromptIdx !== null && updateRule(editingPromptIdx, { ai_decision_prompt: e.target.value })}
                        />
                        <label>动作提示词</label>
                        <textarea
                            rows={7}
                            value={promptRule.action_prompt}
                            onChange={(e) => editingPromptIdx !== null && updateRule(editingPromptIdx, { action_prompt: e.target.value })}
                        />
                        <label>回复提示词</label>
                        <textarea
                            rows={7}
                            value={promptRule.reply_prompt}
                            onChange={(e) => editingPromptIdx !== null && updateRule(editingPromptIdx, { reply_prompt: e.target.value })}
                        />
                    </div>
                ) : (
                    <div />
                )}
            </FormModal>
        </div>
    );
}

function AutoReplyForm({ value, onChange }: { value: AutoReplyConfig; onChange: (next: AutoReplyConfig) => void }) {
    const [openBasicModal, setOpenBasicModal] = useState(false);

    return (
        <div>
            <div className="summary-card">
                <div className="row between wrap gap-8">
                    <strong>基础参数</strong>
                    <button onClick={() => setOpenBasicModal(true)}>编辑基础参数</button>
                </div>
                <div className="rule-preview">
                    模型：{value.model || '未设置'} | 决策模型：{value.decision_model || '未设置'} | 温度：{value.temperature}
                </div>
                <div className="rule-preview">
                    上下文：{value.context_history_limit} 条 / {value.context_max_chars} 字 | 冷却：{value.min_reply_interval_seconds} 秒
                </div>
            </div>

            <FormModal open={openBasicModal} title="自动回复 Agent 基础参数" onClose={() => setOpenBasicModal(false)}>
                <div className="grid2">
                    <div>
                        <label>回复模型</label>
                        <input value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })} />
                    </div>
                    <div>
                        <label>决策模型（可选）</label>
                        <input
                            value={value.decision_model || ''}
                            onChange={(e) => onChange({ ...value, decision_model: e.target.value })}
                        />
                    </div>
                    <div>
                        <label>温度</label>
                        <input
                            type="number"
                            step="0.1"
                            value={value.temperature}
                            onChange={(e) => onChange({ ...value, temperature: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>上下文保留条数</label>
                        <input
                            type="number"
                            value={value.context_history_limit}
                            onChange={(e) => onChange({ ...value, context_history_limit: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>上下文最大字数</label>
                        <input
                            type="number"
                            value={value.context_max_chars}
                            onChange={(e) => onChange({ ...value, context_max_chars: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>上下文时间窗口（秒）</label>
                        <input
                            type="number"
                            value={value.context_window_seconds}
                            onChange={(e) => onChange({ ...value, context_window_seconds: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>最小回复间隔（秒）</label>
                        <input
                            type="number"
                            value={value.min_reply_interval_seconds}
                            onChange={(e) => onChange({ ...value, min_reply_interval_seconds: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>队列刷新间隔（秒）</label>
                        <input
                            type="number"
                            value={value.flush_check_interval_seconds}
                            onChange={(e) => onChange({ ...value, flush_check_interval_seconds: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>待处理过期时间（秒）</label>
                        <input
                            type="number"
                            value={value.pending_expire_seconds}
                            onChange={(e) => onChange({ ...value, pending_expire_seconds: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>待处理最大消息数</label>
                        <input
                            type="number"
                            value={value.pending_max_messages}
                            onChange={(e) => onChange({ ...value, pending_max_messages: Number(e.target.value) })}
                        />
                    </div>
                    <SwitchField
                        checked={value.bypass_cooldown_when_at_bot}
                        onChange={(bypass_cooldown_when_at_bot) => onChange({ ...value, bypass_cooldown_when_at_bot })}
                        label="@机器人时跳过冷却"
                    />
                </div>
            </FormModal>

            <div className="top-gap">
                <label>规则列表</label>
                <AutoReplyRulesEditor rules={value.rules} onChange={(rules) => onChange({ ...value, rules })} />
            </div>
        </div>
    );
}

function DidaForm({ value, onChange }: { value: DidaConfig; onChange: (next: DidaConfig) => void }) {
    const [openBasicModal, setOpenBasicModal] = useState(false);
    const [openGlobalModal, setOpenGlobalModal] = useState(false);

    return (
        <div>
            <div className="summary-card">
                <div className="row between wrap gap-8">
                    <strong>基础参数</strong>
                    <div className="row gap-8">
                        <button onClick={() => setOpenGlobalModal(true)}>编辑 Dida 全局设置</button>
                        <button onClick={() => setOpenBasicModal(true)}>编辑 Agent 参数</button>
                    </div>
                </div>
                <div className="rule-preview">
                    模型：{value.model || '未设置'} | 决策模型：{value.decision_model || '未设置'} | 温度：{value.temperature}
                </div>
                <div className="rule-preview">
                    机器人QQ：{value.bot_qq || '未设置'} | 管理员数：{value.admin_qqs.length} | 冷却：{value.min_reply_interval_seconds} 秒
                </div>
                <div className="rule-preview">
                    OAuth：client_id {value.client_id ? '已配置' : '未配置'} | client_secret {value.client_secret ? '已配置' : '未配置'}
                </div>
                <div className="rule-preview">
                    due_window：{value.due_window_seconds}s | compensate：{value.compensate_window_seconds}s | max_scan：{value.max_tasks_scan_per_user}
                </div>
                <div className="rule-preview">
                    reminder_groups：{value.reminder_group_ids.length} | project_ids：{value.project_ids.length}
                </div>
            </div>

            <FormModal open={openGlobalModal} title="Dida 全局设置" onClose={() => setOpenGlobalModal(false)}>
                <>
                    <div className="grid2">
                        <div>
                            <label>client_id</label>
                            <input
                                value={value.client_id}
                                onChange={(e) => onChange({ ...value, client_id: e.target.value })}
                            />
                        </div>
                        <div>
                            <label>client_secret</label>
                            <input
                                value={value.client_secret}
                                onChange={(e) => onChange({ ...value, client_secret: e.target.value })}
                            />
                        </div>
                        <div className="full-row">
                            <label>redirect_uri</label>
                            <input
                                value={value.redirect_uri}
                                onChange={(e) => onChange({ ...value, redirect_uri: e.target.value })}
                            />
                        </div>
                        <div>
                            <label>due_window_seconds</label>
                            <input
                                type="number"
                                value={value.due_window_seconds}
                                onChange={(e) => onChange({ ...value, due_window_seconds: Number(e.target.value) })}
                            />
                        </div>
                        <div>
                            <label>compensate_window_seconds</label>
                            <input
                                type="number"
                                value={value.compensate_window_seconds}
                                onChange={(e) => onChange({ ...value, compensate_window_seconds: Number(e.target.value) })}
                            />
                        </div>
                        <div>
                            <label>max_tasks_scan_per_user</label>
                            <input
                                type="number"
                                value={value.max_tasks_scan_per_user}
                                onChange={(e) => onChange({ ...value, max_tasks_scan_per_user: Number(e.target.value) })}
                            />
                        </div>
                        <div className="full-row">
                            <label>reminder_group_ids</label>
                            <StringListEditor
                                values={value.reminder_group_ids}
                                onChange={(reminder_group_ids) => onChange({ ...value, reminder_group_ids })}
                            />
                        </div>
                        <div className="full-row">
                            <label>project_ids</label>
                            <StringListEditor
                                values={value.project_ids}
                                onChange={(project_ids) => onChange({ ...value, project_ids })}
                            />
                        </div>
                    </div>
                    <div className="muted top-gap">
                        说明：为兼容当前底层，此处保存时会同步回写到 dida_config。轮询触发频率请在 Scheduler 页面配置 dida.poll 的时间表达式。
                    </div>
                </>
            </FormModal>

            <FormModal open={openBasicModal} title="滴答 Agent 参数" onClose={() => setOpenBasicModal(false)}>
                <div className="grid2">
                    <div>
                        <label>回复模型</label>
                        <input value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })} />
                    </div>
                    <div>
                        <label>决策模型（可选）</label>
                        <input
                            value={value.decision_model || ''}
                            onChange={(e) => onChange({ ...value, decision_model: e.target.value })}
                        />
                    </div>
                    <div>
                        <label>温度</label>
                        <input
                            type="number"
                            step="0.1"
                            value={value.temperature}
                            onChange={(e) => onChange({ ...value, temperature: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>机器人 QQ</label>
                        <input value={value.bot_qq} onChange={(e) => onChange({ ...value, bot_qq: e.target.value })} />
                    </div>
                    <div className="full-row">
                        <label>管理员 QQ 列表</label>
                        <StringListEditor values={value.admin_qqs} onChange={(admin_qqs) => onChange({ ...value, admin_qqs })} />
                    </div>
                    <div>
                        <label>上下文保留条数</label>
                        <input
                            type="number"
                            value={value.context_history_limit}
                            onChange={(e) => onChange({ ...value, context_history_limit: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>上下文最大字数</label>
                        <input
                            type="number"
                            value={value.context_max_chars}
                            onChange={(e) => onChange({ ...value, context_max_chars: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>上下文时间窗口（秒）</label>
                        <input
                            type="number"
                            value={value.context_window_seconds}
                            onChange={(e) => onChange({ ...value, context_window_seconds: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>最小回复间隔（秒）</label>
                        <input
                            type="number"
                            value={value.min_reply_interval_seconds}
                            onChange={(e) => onChange({ ...value, min_reply_interval_seconds: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>队列刷新间隔（秒）</label>
                        <input
                            type="number"
                            value={value.flush_check_interval_seconds}
                            onChange={(e) => onChange({ ...value, flush_check_interval_seconds: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>待处理过期时间（秒）</label>
                        <input
                            type="number"
                            value={value.pending_expire_seconds}
                            onChange={(e) => onChange({ ...value, pending_expire_seconds: Number(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label>待处理最大消息数</label>
                        <input
                            type="number"
                            value={value.pending_max_messages}
                            onChange={(e) => onChange({ ...value, pending_max_messages: Number(e.target.value) })}
                        />
                    </div>
                    <SwitchField
                        checked={value.bypass_cooldown_when_at_bot}
                        onChange={(bypass_cooldown_when_at_bot) => onChange({ ...value, bypass_cooldown_when_at_bot })}
                        label="@机器人时跳过冷却"
                    />
                </div>
            </FormModal>

            <div className="top-gap">
                <label>规则列表</label>
                <DidaRulesEditor rules={value.rules} onChange={(rules) => onChange({ ...value, rules })} />
            </div>
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
    onChange: (next: JsonObject) => void;
}) {
    const schema = getActionSchema(action);
    const [draft, setDraft] = useState(JSON.stringify(params, null, 2));
    const [jsonError, setJsonError] = useState('');

    useEffect(() => {
        setDraft(JSON.stringify(params, null, 2));
    }, [params]);

    return (
        <div className="node">
            <strong>参数</strong>
            {schema ? (
                <div className="grid2 top-gap">
                    {schema.fields.map((field) => {
                        if (field.type === 'boolean') {
                            return (
                                <label key={field.key} className="row gap-8">
                                    <input
                                        type="checkbox"
                                        checked={toBool(params[field.key], false)}
                                        onChange={(e) => onChange({ ...params, [field.key]: e.target.checked })}
                                    />
                                    {field.label}
                                </label>
                            );
                        }

                        if (field.type === 'string[]') {
                            return (
                                <div key={field.key} className="full-row">
                                    <label>{field.label}</label>
                                    <StringListEditor
                                        values={toStringArray(params[field.key])}
                                        onChange={(next) => onChange({ ...params, [field.key]: next })}
                                    />
                                </div>
                            );
                        }

                        if (field.options && field.options.length > 0) {
                            return (
                                <div key={field.key}>
                                    <label>{field.label}</label>
                                    <select
                                        value={toStringValue(params[field.key], toStringValue(field.defaultValue, ''))}
                                        onChange={(e) =>
                                            onChange({
                                                ...params,
                                                [field.key]: e.target.value,
                                            })
                                        }
                                    >
                                        {field.options.map((option) => (
                                            <option key={option.value} value={option.value}>
                                                {option.label}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            );
                        }

                        return (
                            <div key={field.key}>
                                <label>{field.label}</label>
                                <input
                                    type={field.type === 'number' ? 'number' : 'text'}
                                    value={
                                        field.type === 'number'
                                            ? String(toNumber(params[field.key], toNumber(field.defaultValue, 0)))
                                            : toStringValue(params[field.key])
                                    }
                                    onChange={(e) =>
                                        onChange({
                                            ...params,
                                            [field.key]: field.type === 'number' ? Number(e.target.value) : e.target.value,
                                        })
                                    }
                                />
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div className="muted top-gap">该 action 未配置字段 schema，将使用高级 JSON 编辑。</div>
            )}

            <details className="top-gap">
                <summary>高级模式(JSON)</summary>
                <textarea
                    rows={8}
                    value={draft}
                    onChange={(e) => {
                        const nextDraft = e.target.value;
                        setDraft(nextDraft);
                        try {
                            const parsed = JSON.parse(nextDraft) as JsonObject;
                            if (typeof parsed !== 'object' || Array.isArray(parsed) || parsed === null) {
                                setJsonError('params 必须是 JSON 对象');
                                return;
                            }
                            setJsonError('');
                            onChange(parsed);
                        } catch {
                            setJsonError('JSON 格式错误');
                        }
                    }}
                />
                {jsonError && <div className="muted">{jsonError}</div>}
            </details>
        </div>
    );
}

function StepTreeEditor({
    nodes,
    onChange,
    depth = 0,
}: {
    nodes: StepNode[];
    onChange: (next: StepNode[]) => void;
    depth?: number;
}) {
    const updateAt = (index: number, nextNode: StepNode) => {
        const next = [...nodes];
        next[index] = nextNode;
        onChange(next);
    };

    const removeAt = (index: number) => {
        const next = [...nodes];
        next.splice(index, 1);
        onChange(next);
    };

    const move = (index: number, direction: -1 | 1) => {
        const target = index + direction;
        if (target < 0 || target >= nodes.length) return;
        const next = [...nodes];
        const item = next[index];
        next[index] = next[target];
        next[target] = item;
        onChange(next);
    };

    return (
        <div className="schedule-grid" style={{ marginLeft: depth * 12 }}>
            {nodes.map((node, idx) => (
                <div key={node.id || `${node.kind}_${idx}`} className="node">
                    <div className="row between gap-8 wrap">
                        <strong>
                            {node.kind === 'action' ? 'Action' : 'Group'} #{idx + 1}
                        </strong>
                        <div className="row gap-8 wrap">
                            <button onClick={() => move(idx, -1)} disabled={idx === 0}>
                                上移
                            </button>
                            <button onClick={() => move(idx, 1)} disabled={idx === nodes.length - 1}>
                                下移
                            </button>
                            <button onClick={() => removeAt(idx)}>删除</button>
                        </div>
                    </div>

                    <div className="grid2 top-gap">
                        <div>
                            <label>节点类型</label>
                            <select
                                value={node.kind}
                                onChange={(e) => updateAt(idx, defaultStep(e.target.value as StepNode['kind']))}
                            >
                                <option value="action">action</option>
                                <option value="group">group</option>
                            </select>
                        </div>
                    </div>

                    {node.kind === 'action' && (
                        <>
                            <div>
                                <label>action</label>
                                <select
                                    value={node.action || 'core.send_group_msg'}
                                    onChange={(e) =>
                                        updateAt(idx, {
                                            ...node,
                                            action: e.target.value,
                                            params: withActionDefaults(e.target.value, {}, true),
                                        })
                                    }
                                >
                                    {ACTION_OPTIONS.map((item) => (
                                        <option key={item.value} value={item.value}>
                                            {item.label}
                                        </option>
                                    ))}
                                    {!ACTION_OPTIONS.find((item) => item.value === node.action) && node.action && (
                                        <option value={node.action}>{node.action}</option>
                                    )}
                                </select>
                            </div>
                            <ActionParamsEditor
                                action={node.action || 'core.send_group_msg'}
                                params={withActionDefaults(node.action || 'core.send_group_msg', node.params || {}, true)}
                                onChange={(nextParams) =>
                                    updateAt(idx, {
                                        ...node,
                                        params: withActionDefaults(node.action || 'core.send_group_msg', nextParams, true),
                                    })
                                }
                            />
                        </>
                    )}

                    {node.kind === 'group' && (
                        <>
                            <div>
                                <label>group 名称</label>
                                <input
                                    value={node.name || ''}
                                    onChange={(e) => updateAt(idx, { ...node, name: e.target.value })}
                                />
                            </div>
                            <div className="top-gap">
                                <label>children</label>
                                <StepTreeEditor
                                    nodes={node.children || []}
                                    depth={depth + 1}
                                    onChange={(children) => updateAt(idx, { ...node, children })}
                                />
                            </div>
                        </>
                    )}
                </div>
            ))}

            <div className="row gap-8 top-gap wrap">
                <button onClick={() => onChange([...nodes, defaultStep('action')])}>+ Action</button>
                <button onClick={() => onChange([...nodes, defaultStep('group')])}>+ Group</button>
            </div>
        </div>
    );
}

function normalizeTimeUnit(value: string, max: number): string {
    const n = Number(value);
    if (!Number.isFinite(n)) return '00';
    const safe = Math.max(0, Math.min(max, Math.floor(n)));
    return String(safe).padStart(2, '0');
}

function cronFromHms(hour: string, minute: string, second: string): string {
    const hh = normalizeTimeUnit(hour, 23);
    const mm = normalizeTimeUnit(minute, 59);
    const ss = normalizeTimeUnit(second, 59);
    return `${ss} ${mm} ${hh} * * *`;
}

function hmsFromCron(expression: string | undefined): { hour: string; minute: string; second: string } {
    const raw = (expression || '').trim();
    if (!raw) return { hour: '08', minute: '00', second: '00' };
    const parts = raw.split(/\s+/);
    if (parts.length >= 6) {
        return {
            second: normalizeTimeUnit(parts[0], 59),
            minute: normalizeTimeUnit(parts[1], 59),
            hour: normalizeTimeUnit(parts[2], 23),
        };
    }
    if (parts.length >= 5) {
        return {
            second: '00',
            minute: normalizeTimeUnit(parts[0], 59),
            hour: normalizeTimeUnit(parts[1], 23),
        };
    }
    return { hour: '08', minute: '00', second: '00' };
}

function StepTreePreview({ nodes }: { nodes: StepNode[] }) {
    if (!nodes || nodes.length === 0) {
        return <div className="muted">暂无步骤</div>;
    }

    return (
        <ul className="step-preview-tree">
            {nodes.map((node, idx) => (
                <li key={node.id || `${node.kind}_${idx}`}>
                    <span>
                        {node.kind === 'group'
                            ? `分组: ${node.name || '未命名分组'}`
                            : `动作: ${node.action || '未设置动作'}`}
                    </span>
                    {node.kind === 'group' && (node.children || []).length > 0 && (
                        <StepTreePreview nodes={node.children || []} />
                    )}
                </li>
            ))}
        </ul>
    );
}

export default function App() {
    const [tab, setTab] = useState<TabKey>('basic');
    const [openSchedulerHelp, setOpenSchedulerHelp] = useState(false);
    const [status, setStatus] = useState('idle');
    const [error, setError] = useState('');

    const [llmApiKey, setLlmApiKey] = useState('');
    const [llmApiBaseUrl, setLlmApiBaseUrl] = useState('');

    const [agent, setAgent] = useState<AgentConfigRoot>({});
    const [selectedAgent, setSelectedAgent] = useState<FixedAgentKey>('summary_config');

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
        restart_policy: 'sudo-docker-restart',
    });
    const [deployLogs, setDeployLogs] = useState<string[]>([]);
    const [snapshots, setSnapshots] = useState<string[]>([]);
    const [selectedScheduleIndex, setSelectedScheduleIndex] = useState(0);
    const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
    const [timelineCount, setTimelineCount] = useState(20);
    const [schedulerMessage, setSchedulerMessage] = useState('');
    const [openScheduleBasicModal, setOpenScheduleBasicModal] = useState(false);
    const [openScheduleStepModal, setOpenScheduleStepModal] = useState(false);

    const fixedSections = useMemo(() => normalizeFixedSections(agent), [agent]);
    const schedulerConfig = useMemo(() => getSchedulerManagerConfig(agent), [agent]);
    const schedulerManagerEnabled = useMemo(() => isSchedulerManagerEnabled(agent), [agent]);
    const hasEnabledSummaryDailyReport = useMemo(
        () =>
            schedulerConfig.schedules.some(
                (item) => item.enabled && hasActionInStepNodes(item.steps_tree, 'summary.daily_report'),
            ),
        [schedulerConfig.schedules],
    );
    const showSummaryMissingWarning = schedulerManagerEnabled && !hasEnabledSummaryDailyReport;

    useEffect(() => {
        (async () => {
            try {
                setStatus('loading');
                const all = await getAllConfig();
                setLlmApiKey(all.env?.LLM_API_KEY || '');
                setLlmApiBaseUrl(all.env?.LLM_API_BASE_URL || '');
                setAgent(all.agent || {});
                const history = await listSnapshots();
                setSnapshots(history.snapshots || []);
                setStatus('ready');
            } catch (e) {
                setStatus('error');
                setError(String(e));
            }
        })();
    }, []);

    useEffect(() => {
        const timer = setTimeout(() => {
            if (status !== 'ready') return;
            saveEnv(llmApiKey, llmApiBaseUrl).catch((e) => setError(String(e)));
        }, 300);
        return () => clearTimeout(timer);
    }, [llmApiKey, llmApiBaseUrl, status]);

    useEffect(() => {
        if (schedulerConfig.schedules.length === 0) {
            setSelectedScheduleIndex(0);
            return;
        }
        if (selectedScheduleIndex > schedulerConfig.schedules.length - 1) {
            setSelectedScheduleIndex(schedulerConfig.schedules.length - 1);
        }
    }, [schedulerConfig.schedules.length, selectedScheduleIndex]);

    const updateFixedSectionConfig = (key: FixedAgentKey, cfg: JsonObject) => {
        const next = cloneAgent(agent);
        next[key] = {
            file_name: fixedSections[key].file_name,
            config: cfg,
        };
        setAgent(next);
    };

    const updateDidaAgentConfig = (cfg: DidaConfig) => {
        const next = cloneAgent(agent);
        // 更新 dida_agent_config (dida_agent.py)
        next.dida_agent_config = {
            file_name: fixedSections.dida_agent_config.file_name,
            config: buildDidaAgentPersistedConfig(cfg as unknown as JsonObject),
        };
        // 同步更新 dida_config (dida_scheduler.py)，共享 OAuth 基础参数
        next.dida_config = {
            file_name: fixedSections.dida_config.file_name,
            config: {
                client_id: cfg.client_id,
                client_secret: cfg.client_secret,
                redirect_uri: cfg.redirect_uri,
                due_window_seconds: cfg.due_window_seconds,
                compensate_window_seconds: cfg.compensate_window_seconds,
                max_tasks_scan_per_user: cfg.max_tasks_scan_per_user,
                reminder_group_ids: cfg.reminder_group_ids,
                project_ids: cfg.project_ids,
            },
        };
        setAgent(next);
    };

    const updateSchedulerConfig = (nextConfig: SchedulerManagerConfig) => {
        const next = cloneAgent(agent);
        next.scheduler_manager = {
            file_name: toStringValue(asObject(next.scheduler_manager).file_name, 'scheduler_manager.py'),
            config: {
                timezone: nextConfig.timezone,
                schedules: nextConfig.schedules,
            },
        };
        setAgent(next);
    };

    const updateScheduleAt = (index: number, nextSchedule: ScheduleConfig) => {
        const nextSchedules = [...schedulerConfig.schedules];
        nextSchedules[index] = nextSchedule;
        updateSchedulerConfig({ ...schedulerConfig, schedules: nextSchedules });
    };

    const moveSchedule = (index: number, direction: -1 | 1) => {
        const target = index + direction;
        if (target < 0 || target >= schedulerConfig.schedules.length) return;
        const next = [...schedulerConfig.schedules];
        const item = next[index];
        next[index] = next[target];
        next[target] = item;
        updateSchedulerConfig({ ...schedulerConfig, schedules: next });
        setSelectedScheduleIndex(target);
    };

    const addSchedule = () => {
        const next = [...schedulerConfig.schedules, defaultSchedule()];
        updateSchedulerConfig({ ...schedulerConfig, schedules: next });
        setSelectedScheduleIndex(next.length - 1);
    };

    const duplicateSchedule = (index: number) => {
        const item = schedulerConfig.schedules[index];
        if (!item) return;
        const clone = JSON.parse(JSON.stringify(item)) as ScheduleConfig;
        clone.name = `${item.name}_copy`;
        const next = [...schedulerConfig.schedules];
        next.splice(index + 1, 0, clone);
        updateSchedulerConfig({ ...schedulerConfig, schedules: next });
        setSelectedScheduleIndex(index + 1);
    };

    const removeSchedule = (index: number) => {
        const next = [...schedulerConfig.schedules];
        next.splice(index, 1);
        updateSchedulerConfig({ ...schedulerConfig, schedules: next });
        setSelectedScheduleIndex(Math.max(0, Math.min(index, next.length - 1)));
    };

    const saveSchedulerConfig = async () => {
        try {
            setSchedulerMessage('保存中...');
            const compiled = await compileSchedules(schedulerConfig.schedules);
            const compiledConfig: SchedulerManagerConfig = {
                timezone: schedulerConfig.timezone,
                schedules: compiled.schedules,
            };
            const payload = cloneAgent(agent);
            payload.scheduler_manager = {
                file_name: toStringValue(asObject(payload.scheduler_manager).file_name, 'scheduler_manager.py'),
                config: compiledConfig,
            };
            await saveAgent(payload as Record<string, unknown>);
            setAgent(payload);
            setStatus('saved');
            setSchedulerMessage('Scheduler 已保存并完成编译');
        } catch (e) {
            setError(String(e));
            setSchedulerMessage('保存失败');
        }
    };

    const loadTimeline = async () => {
        try {
            const resp = await getTimeline(
                schedulerConfig.timezone,
                schedulerConfig.schedules,
                Math.max(1, Math.min(200, timelineCount)),
            );
            setTimelineEvents(resp.events || []);
            setSchedulerMessage(`时间线已刷新，共 ${resp.events?.length || 0} 条`);
        } catch (e) {
            setError(String(e));
        }
    };

    const saveAgentFull = async () => {
        try {
            const payload = cloneAgent(agent);
            for (const fixedKey of FIXED_AGENT_OPTIONS.map((x) => x.key)) {
                payload[fixedKey] = {
                    file_name: fixedSections[fixedKey].file_name,
                    config: fixedSections[fixedKey].config,
                };
            }

            // 特殊处理 dida：合并 dida_config 和 dida_agent_config 的冗余字段至 dida_config
            const didaAgentCfg = asObject(asObject(payload.dida_agent_config).config);
            const didaCfg = asObject(asObject(payload.dida_config).config);

            const mergedDidaCfg = {
                ...didaCfg,
                client_id: toStringValue(didaCfg.client_id, toStringValue(didaAgentCfg.client_id, '')),
                client_secret: toStringValue(didaCfg.client_secret, toStringValue(didaAgentCfg.client_secret, '')),
                redirect_uri: toStringValue(didaCfg.redirect_uri, toStringValue(didaAgentCfg.redirect_uri, '')),
                due_window_seconds: toNumber(didaCfg.due_window_seconds, toNumber(didaAgentCfg.due_window_seconds, 60)),
                compensate_window_seconds: toNumber(didaCfg.compensate_window_seconds, 0),
                max_tasks_scan_per_user: toNumber(didaCfg.max_tasks_scan_per_user, toNumber(didaAgentCfg.max_tasks_scan_per_user, 200)),
                reminder_group_ids: toStringArray(didaCfg.reminder_group_ids),
                project_ids: (() => {
                    const fromConfig = toStringArray(didaCfg.project_ids);
                    if (fromConfig.length > 0) return fromConfig;
                    return toStringArray(didaAgentCfg.project_ids);
                })(),
            };

            payload.dida_agent_config = {
                file_name: toStringValue(asObject(payload.dida_agent_config).file_name, fixedSections.dida_agent_config.file_name),
                config: buildDidaAgentPersistedConfig(didaAgentCfg),
            };
            payload.dida_config = {
                file_name: toStringValue(asObject(payload.dida_config).file_name, fixedSections.dida_config.file_name),
                config: mergedDidaCfg,
            };

            delete payload.dida_scheduler_config;
            await saveAgent(payload as Record<string, unknown>);
            setStatus('saved');
        } catch (e) {
            setError(String(e));
        }
    };

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

    const runPullAndRefresh = async () => {
        try {
            const result = await deployPull({
                ...deployForm,
                pull_env: deployForm.push_env,
                pull_agent_yaml: deployForm.push_agent_yaml,
            });
            const all = await getAllConfig();
            setLlmApiKey(all.env?.LLM_API_KEY || '');
            setLlmApiBaseUrl(all.env?.LLM_API_BASE_URL || '');
            setAgent(all.agent || {});
            setDeployLogs(result.logs || [result.message]);
            setStatus('ready');
        } catch (e) {
            setError(String(e));
        }
    };

    const restore = async (snapshot: string, target: 'env' | 'agent') => {
        try {
            await restoreSnapshot(snapshot, target);
            const all = await getAllConfig();
            setLlmApiKey(all.env?.LLM_API_KEY || '');
            setLlmApiBaseUrl(all.env?.LLM_API_BASE_URL || '');
            setAgent(all.agent || {});
            const history = await listSnapshots();
            setSnapshots(history.snapshots || []);
            setStatus('restored');
        } catch (e) {
            setError(String(e));
        }
    };

    const summaryRulesRaw = Array.isArray(fixedSections.summary_config.config.rules)
        ? fixedSections.summary_config.config.rules
        : [];
    const summaryRules: SummaryRule[] = summaryRulesRaw.map((item) => {
        const obj = asObject(item as JsonValue);
        return {
            enabled: toBool(obj.enabled, true),
            target_type: toStringValue(obj.target_type, 'private') as 'private' | 'group',
            target_id: toStringValue(obj.target_id),
            run_mode: toStringValue(obj.run_mode, 'all') as 'all' | 'auto' | 'manual',
        };
    });

    const summaryConfig: SummaryConfig = {
        model: toStringValue(fixedSections.summary_config.config.model),
        temperature: toNumber(fixedSections.summary_config.config.temperature, 0.2),
        max_line_chars: toNumber(fixedSections.summary_config.config.max_line_chars, 300),
        max_lines: toNumber(fixedSections.summary_config.config.max_lines, 500),
        summary_chat_scope: toStringValue(
            fixedSections.summary_config.config.summary_chat_scope,
            'group',
        ) as SummaryConfig['summary_chat_scope'],
        summary_group_filter_mode: toStringValue(
            fixedSections.summary_config.config.summary_group_filter_mode,
            'all',
        ) as SummaryConfig['summary_group_filter_mode'],
        summary_group_ids: toStringArray(fixedSections.summary_config.config.summary_group_ids),
        summary_global_overview: toBool(fixedSections.summary_config.config.summary_global_overview, true),
        summary_send_mode: toStringValue(
            fixedSections.summary_config.config.summary_send_mode,
            'multi_message',
        ) as SummaryConfig['summary_send_mode'],
        summary_group_reduce_enabled: toBool(
            fixedSections.summary_config.config.summary_group_reduce_enabled,
            true,
        ),
        rules: summaryRules.length > 0 ? summaryRules : [makeDefaultSummaryRule()],
    };

    const forwardConfig: ForwardConfig = {
        model: toStringValue(fixedSections.forward_config.config.model),
        decision_model: toStringValue(fixedSections.forward_config.config.decision_model),
        temperature: toNumber(fixedSections.forward_config.config.temperature, 0),
        monitor_group_qq_number: toStringArray(fixedSections.forward_config.config.monitor_group_qq_number),
        forward_decision_prompt: toStringValue(fixedSections.forward_config.config.forward_decision_prompt),
    };

    const autoReplyRulesRaw = Array.isArray(fixedSections.auto_reply_config.config.rules)
        ? fixedSections.auto_reply_config.config.rules
        : [];
    const autoReplyRules: AutoReplyRule[] = autoReplyRulesRaw.map((item) => {
        const obj = asObject(item as JsonValue);
        return {
            enabled: toBool(obj.enabled, true),
            chat_type: toStringValue(obj.chat_type, 'group') as 'group' | 'private',
            number: toStringValue(obj.number),
            trigger_mode: toStringValue(obj.trigger_mode, 'always'),
            keywords: toStringArray(obj.keywords),
            ai_decision_prompt: toStringValue(obj.ai_decision_prompt),
            reply_prompt: toStringValue(obj.reply_prompt),
            temperature: toNumber(obj.temperature, 0.4),
        };
    });

    const autoReplyConfig: AutoReplyConfig = {
        model: toStringValue(fixedSections.auto_reply_config.config.model),
        decision_model: toStringValue(fixedSections.auto_reply_config.config.decision_model),
        temperature: toNumber(fixedSections.auto_reply_config.config.temperature, 0.4),
        context_history_limit: toNumber(fixedSections.auto_reply_config.config.context_history_limit, 50),
        context_max_chars: toNumber(fixedSections.auto_reply_config.config.context_max_chars, 2000),
        context_window_seconds: toNumber(fixedSections.auto_reply_config.config.context_window_seconds, 0),
        min_reply_interval_seconds: toNumber(
            fixedSections.auto_reply_config.config.min_reply_interval_seconds,
            10,
        ),
        flush_check_interval_seconds: toNumber(
            fixedSections.auto_reply_config.config.flush_check_interval_seconds,
            10,
        ),
        pending_expire_seconds: toNumber(fixedSections.auto_reply_config.config.pending_expire_seconds, 3600),
        bypass_cooldown_when_at_bot: toBool(
            fixedSections.auto_reply_config.config.bypass_cooldown_when_at_bot,
            false,
        ),
        pending_max_messages: toNumber(fixedSections.auto_reply_config.config.pending_max_messages, 50),
        rules: autoReplyRules.length > 0 ? autoReplyRules : [makeDefaultAutoRule()],
    };

    const didaRulesRaw = Array.isArray(fixedSections.dida_agent_config.config.rules)
        ? fixedSections.dida_agent_config.config.rules
        : [];
    const didaRules: DidaRule[] = didaRulesRaw.map((item) => {
        const obj = asObject(item as JsonValue);
        return {
            enabled: toBool(obj.enabled, true),
            chat_type: toStringValue(obj.chat_type, 'group') as 'group' | 'private',
            number: toStringValue(obj.number),
            trigger_mode: toStringValue(obj.trigger_mode, 'ai_decide'),
            dida_enabled: toBool(obj.dida_enabled, true),
            ai_decision_prompt: toStringValue(obj.ai_decision_prompt),
            action_prompt: toStringValue(obj.action_prompt),
            reply_prompt: toStringValue(obj.reply_prompt),
            action_temperature: toNumber(obj.action_temperature, 0),
            reply_temperature: toNumber(obj.reply_temperature, 0.5),
        };
    });

    const didaConfig: DidaConfig = {
        client_id: toStringValue(
            fixedSections.dida_config.config.client_id,
            toStringValue(fixedSections.dida_agent_config.config.client_id),
        ),
        client_secret: toStringValue(
            fixedSections.dida_config.config.client_secret,
            toStringValue(fixedSections.dida_agent_config.config.client_secret),
        ),
        redirect_uri: toStringValue(
            fixedSections.dida_config.config.redirect_uri,
            toStringValue(fixedSections.dida_agent_config.config.redirect_uri),
        ),
        model: toStringValue(fixedSections.dida_agent_config.config.model),
        decision_model: toStringValue(fixedSections.dida_agent_config.config.decision_model),
        temperature: toNumber(fixedSections.dida_agent_config.config.temperature, 0.1),
        bot_qq: toStringValue(fixedSections.dida_agent_config.config.bot_qq),
        admin_qqs: toStringArray(fixedSections.dida_agent_config.config.admin_qqs),
        due_window_seconds: toNumber(
            fixedSections.dida_config.config.due_window_seconds,
            toNumber(fixedSections.dida_agent_config.config.due_window_seconds, 60),
        ),
        compensate_window_seconds: toNumber(
            fixedSections.dida_config.config.compensate_window_seconds,
            0,
        ),
        max_tasks_scan_per_user: toNumber(
            fixedSections.dida_config.config.max_tasks_scan_per_user,
            toNumber(fixedSections.dida_agent_config.config.max_tasks_scan_per_user, 200),
        ),
        reminder_group_ids: toStringArray(fixedSections.dida_config.config.reminder_group_ids),
        project_ids: (() => {
            const fromConfig = toStringArray(fixedSections.dida_config.config.project_ids);
            if (fromConfig.length > 0) return fromConfig;
            return toStringArray(fixedSections.dida_agent_config.config.project_ids);
        })(),
        context_history_limit: toNumber(fixedSections.dida_agent_config.config.context_history_limit, 50),
        context_max_chars: toNumber(fixedSections.dida_agent_config.config.context_max_chars, 2000),
        context_window_seconds: toNumber(fixedSections.dida_agent_config.config.context_window_seconds, 0),
        min_reply_interval_seconds: toNumber(
            fixedSections.dida_agent_config.config.min_reply_interval_seconds,
            10,
        ),
        flush_check_interval_seconds: toNumber(
            fixedSections.dida_agent_config.config.flush_check_interval_seconds,
            10,
        ),
        pending_expire_seconds: toNumber(fixedSections.dida_agent_config.config.pending_expire_seconds, 3600),
        bypass_cooldown_when_at_bot: toBool(
            fixedSections.dida_agent_config.config.bypass_cooldown_when_at_bot,
            true,
        ),
        pending_max_messages: toNumber(fixedSections.dida_agent_config.config.pending_max_messages, 50),
        rules: didaRules.length > 0 ? didaRules : [makeDefaultDidaRule()],
    };

    const didaConfigured =
        didaConfig.client_id.trim().length > 0 &&
        didaConfig.client_secret.trim().length > 0 &&
        didaConfig.redirect_uri.trim().length > 0;
    const hasEnabledDidaPoll = useMemo(
        () =>
            schedulerConfig.schedules.some(
                (item) => item.enabled && hasActionInStepNodes(item.steps_tree, 'dida.poll'),
            ),
        [schedulerConfig.schedules],
    );
    const showDidaPollMissingWarning = schedulerManagerEnabled && didaConfigured && !hasEnabledDidaPoll;

    const selectedSchedule = schedulerConfig.schedules[selectedScheduleIndex];

    return (
        <div className="layout">
            <div className="sidebar">
                <header className="topbar" style={{ background: 'transparent', border: 'none', padding: '0 0 16px', marginBottom: '8px', borderBottom: '1px solid var(--line)', display: 'block', textAlign: 'center' }}>
                    <h2 style={{ margin: 0, fontSize: '18px' }}>Config Studio</h2>
                    <div className="status" style={{ fontSize: '12px', marginTop: '4px' }}>状态: {status}</div>
                </header>

                <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {[
                        ['basic', '基础设置'],
                        ['agent', 'Agent设置'],
                        ['scheduler', 'Scheduler设置'],
                        ['deploy', '推送中心'],
                        ['history', '变更历史'],
                        ['terminal', '控制台'],
                    ].map(([key, label]) => (
                        <button key={key} className={`sidebar-nav-item ${tab === key ? 'active' : ''}`} onClick={() => setTab(key as TabKey)}>
                            {label}
                        </button>
                    ))}
                </nav>
            </div>
            <div className="main-content">

                {error && (
                    <div className="error">
                        {error}
                        <div className="hint">若显示 Failed to fetch，请确认后端 API 已在 8787 端口启动，或前端代理是否可访问。</div>
                    </div>
                )}

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
                    <section className="panel">
                        <div className="row between" style={{ paddingBottom: '12px', borderBottom: '1px solid var(--line)', marginBottom: '16px' }}>
                            <h2 style={{ margin: 0 }}>固定 Agent 表单</h2>
                            <button className="btn-primary" onClick={saveAgentFull}>保存 Agent 配置</button>
                        </div>

                        <div className="agent-tabs">
                            {FIXED_AGENT_OPTIONS.map((option) => (
                                <button
                                    key={option.key}
                                    className={selectedAgent === option.key ? 'active' : ''}
                                    onClick={() => setSelectedAgent(option.key as FixedAgentKey)}
                                >
                                    {option.label}
                                </button>
                            ))}
                        </div>

                        {selectedAgent === 'summary_config' && (
                            <SummaryForm
                                value={summaryConfig}
                                onChange={(next) => updateFixedSectionConfig('summary_config', next as unknown as JsonObject)}
                            />
                        )}

                        {selectedAgent === 'forward_config' && (
                            <ForwardForm
                                value={forwardConfig}
                                onChange={(next) => updateFixedSectionConfig('forward_config', next as unknown as JsonObject)}
                            />
                        )}

                        {selectedAgent === 'auto_reply_config' && (
                            <AutoReplyForm
                                value={autoReplyConfig}
                                onChange={(next) => updateFixedSectionConfig('auto_reply_config', next as unknown as JsonObject)}
                            />
                        )}

                        {selectedAgent === 'dida_agent_config' && (
                            <DidaForm
                                value={didaConfig}
                                onChange={(next) => updateDidaAgentConfig(next)}
                            />
                        )}


                    </section>
                )}

                {tab === 'scheduler' && (
                    <section className="panel">
                        <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            Scheduler 设置
                            <button
                                className="icon-btn"
                                onClick={() => setOpenSchedulerHelp(true)}
                                title="使用说明"
                                style={{ borderRadius: '50%', width: '24px', height: '24px', padding: 0, minWidth: 0, lineHeight: '24px', display: 'inline-block', textAlign: 'center', fontSize: '14px', cursor: 'pointer' }}
                            >
                                ?
                            </button>
                        </h2>
                        {showSummaryMissingWarning && (
                            <div className="warning top-gap">
                                已开启 scheduler_manager，但未发现启用的 summary.daily_report 任务；日报不会自动执行。
                            </div>
                        )}
                        {showDidaPollMissingWarning && (
                            <div className="warning top-gap">
                                已配置 Dida 清单并开启 scheduler_manager，但未发现启用的 dida.poll 任务；Dida 清单不会自动拉取。
                            </div>
                        )}


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
                            <div className="schedule-grid">
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
                                <FormModal
                                    open={openScheduleBasicModal}
                                    title="Schedule 基础参数"
                                    onClose={() => setOpenScheduleBasicModal(false)}
                                >
                                    <div className="grid2">
                                        <div>
                                            <label>任务名称</label>
                                            <input
                                                value={selectedSchedule.name}
                                                onChange={(e) =>
                                                    updateScheduleAt(selectedScheduleIndex, {
                                                        ...selectedSchedule,
                                                        name: e.target.value,
                                                    })
                                                }
                                            />
                                        </div>
                                        <div>
                                            <label>任务类型</label>
                                            <select
                                                value={selectedSchedule.type}
                                                onChange={(e) => {
                                                    const nextType = e.target.value as 'cron' | 'interval';
                                                    const nextSchedule: ScheduleConfig = {
                                                        ...selectedSchedule,
                                                        type: nextType,
                                                    };
                                                    if (nextType === 'cron') {
                                                        const hms = hmsFromCron(selectedSchedule.expression);
                                                        nextSchedule.expression = cronFromHms(hms.hour, hms.minute, hms.second);
                                                        delete nextSchedule.seconds;
                                                    } else {
                                                        nextSchedule.seconds = selectedSchedule.seconds || 60;
                                                        delete nextSchedule.expression;
                                                    }
                                                    updateScheduleAt(selectedScheduleIndex, nextSchedule);
                                                }}
                                            >
                                                <option value="cron">定时(Cron)</option>
                                                <option value="interval">间隔(Interval)</option>
                                            </select>
                                        </div>

                                        <SwitchField
                                            checked={selectedSchedule.enabled}
                                            onChange={(enabled) =>
                                                updateScheduleAt(selectedScheduleIndex, {
                                                    ...selectedSchedule,
                                                    enabled,
                                                })
                                            }
                                            label="启用任务"
                                        />

                                        {selectedSchedule.type === 'cron' ? (
                                            <div className="full-row">
                                                <label>执行时间（24小时制）</label>
                                                <div className="row gap-8 wrap">
                                                    {(() => {
                                                        const hms = hmsFromCron(selectedSchedule.expression);
                                                        return (
                                                            <>
                                                                <input
                                                                    className="time-field"
                                                                    value={hms.hour}
                                                                    onChange={(e) =>
                                                                        updateScheduleAt(selectedScheduleIndex, {
                                                                            ...selectedSchedule,
                                                                            expression: cronFromHms(e.target.value, hms.minute, hms.second),
                                                                        })
                                                                    }
                                                                    placeholder="小时"
                                                                />
                                                                <span>:</span>
                                                                <input
                                                                    className="time-field"
                                                                    value={hms.minute}
                                                                    onChange={(e) =>
                                                                        updateScheduleAt(selectedScheduleIndex, {
                                                                            ...selectedSchedule,
                                                                            expression: cronFromHms(hms.hour, e.target.value, hms.second),
                                                                        })
                                                                    }
                                                                    placeholder="分钟"
                                                                />
                                                                <span>:</span>
                                                                <input
                                                                    className="time-field"
                                                                    value={hms.second}
                                                                    onChange={(e) =>
                                                                        updateScheduleAt(selectedScheduleIndex, {
                                                                            ...selectedSchedule,
                                                                            expression: cronFromHms(hms.hour, hms.minute, e.target.value),
                                                                        })
                                                                    }
                                                                    placeholder="秒"
                                                                />
                                                            </>
                                                        );
                                                    })()}
                                                </div>
                                                <div className="muted">系统会在后台自动转换为 cron 表达式并保存。</div>
                                            </div>
                                        ) : (
                                            <div>
                                                <label>间隔秒数</label>
                                                <input
                                                    type="number"
                                                    value={selectedSchedule.seconds || 60}
                                                    onChange={(e) =>
                                                        updateScheduleAt(selectedScheduleIndex, {
                                                            ...selectedSchedule,
                                                            seconds: Number(e.target.value),
                                                        })
                                                    }
                                                />
                                            </div>
                                        )}
                                    </div>
                                </FormModal>

                                <FormModal
                                    open={openScheduleStepModal}
                                    title="Schedule 执行步骤"
                                    onClose={() => setOpenScheduleStepModal(false)}
                                >
                                    <div>
                                        <label>步骤树</label>
                                        <StepTreeEditor
                                            nodes={selectedSchedule.steps_tree || []}
                                            onChange={(steps_tree) =>
                                                updateScheduleAt(selectedScheduleIndex, {
                                                    ...selectedSchedule,
                                                    steps_tree,
                                                })
                                            }
                                        />
                                    </div>
                                </FormModal>
                            </>
                        ) : (
                            <div className="muted top-gap">暂无 Schedule，请先新增。</div>
                        )}
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
                                <button onClick={runPullAndRefresh}>拉取并刷新</button>
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
                        <h3>快照历史</h3>
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
                        <FormModal open={openSchedulerHelp} title="Scheduler 功能使用说明" onClose={() => setOpenSchedulerHelp(false)}>
                            <div className="scheduler-help-content" style={{ maxHeight: '600px', overflowY: 'auto' }}>

                                <h4>1. 可用参数</h4>
                                <ul>
                                    <li>timezone: 调度时区，例如 Asia/Shanghai。cron 计算会使用该时区。</li>
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
                        </FormModal>
                    </section>
                )}

                {tab === 'terminal' && (
                    <section className="panel card" style={{ padding: 0 }}>
                        <TerminalPanel deployConfig={deployForm} />
                    </section>
                )}
            </div>
        </div>
    );
}
