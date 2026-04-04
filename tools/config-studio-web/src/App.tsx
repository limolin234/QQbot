import { useEffect, useMemo, useState } from 'react';
import {
    compileSchedules,
    deployPush,
    getAllConfig,
    getTimeline,
    listSnapshots,
    restoreSnapshot,
    saveAgent,
    saveEnv,
    testConnection
} from './api';
import type {
    AgentConfigRoot,
    JsonObject,
    JsonValue,
    ScheduleConfig,
    StepNode,
    TabKey,
    TimelineEvent
} from './types';

type FixedAgentKey = 'summary_config' | 'forward_config' | 'auto_reply_config' | 'dida_agent_config';

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
    model: string;
    decision_model?: string;
    temperature: number;
    bot_qq: string;
    admin_qqs: string[];
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
};

type ActionSchema = {
    action: string;
    label: string;
    fields: ActionFieldSchema[];
};

const ACTION_SCHEMAS: ActionSchema[] = [
    {
        action: 'core.send_group_msg',
        label: '发送群消息',
        fields: [
            { key: 'group_id', label: '群号', type: 'string' },
            { key: 'message', label: '消息内容', type: 'string' },
        ],
    },
    {
        action: 'summary.daily_report',
        label: '每日总结',
        fields: [{ key: 'run_mode', label: '运行模式', type: 'string' }],
    },
    {
        action: 'dida.poll',
        label: '轮询任务',
        fields: [{ key: 'project_ids', label: '项目 ID 列表', type: 'string[]' }],
    },
    {
        action: 'dida.push_task_list',
        label: '推送任务清单',
        fields: [
            { key: 'group_id', label: '群号', type: 'string' },
            { key: 'user_qq', label: '用户 QQ', type: 'string' },
            { key: 'day_range', label: '时间范围', type: 'string' },
            { key: 'limit', label: '条数上限', type: 'number' },
        ],
    },
];

const ACTION_OPTIONS = ACTION_SCHEMAS.map((item) => ({ value: item.action, label: `${item.action} (${item.label})` }));

type FixedSection = {
    file_name: string;
    config: JsonObject;
};

const FIXED_AGENT_OPTIONS: Array<{ key: FixedAgentKey; label: string }> = [
    { key: 'summary_config', label: 'Summary Agent' },
    { key: 'forward_config', label: 'Forward Agent' },
    { key: 'auto_reply_config', label: 'Auto Reply Agent' },
    { key: 'dida_agent_config', label: 'Dida Agent' },
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

function makeNodeId(): string {
    return `node_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function getActionSchema(action: string): ActionSchema | undefined {
    return ACTION_SCHEMAS.find((item) => item.action === action);
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
        node.params = asObject(obj.params);
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
                return {
                    id: makeNodeId(),
                    kind: 'action',
                    action: toStringValue(legacyObj.action, 'core.send_group_msg'),
                    params: asObject(legacyObj.params),
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
    return {
        id: makeNodeId(),
        kind: 'action',
        action: 'core.send_group_msg',
        params: { group_id: '', message: '' },
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
    };
}

function normalizeFixedSections(agent: AgentConfigRoot): Record<FixedAgentKey, FixedSection> {
    const defaults = defaultSections();
    for (const key of FIXED_AGENT_OPTIONS.map((x) => x.key)) {
        const rawSection = asObject(agent[key]);
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

function SummaryForm({ value, onChange }: { value: SummaryConfig; onChange: (next: SummaryConfig) => void }) {
    return (
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
            <label className="row gap-8">
                <input
                    type="checkbox"
                    checked={value.summary_global_overview}
                    onChange={(e) => onChange({ ...value, summary_global_overview: e.target.checked })}
                />
                summary_global_overview
            </label>
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
            <label className="row gap-8">
                <input
                    type="checkbox"
                    checked={value.summary_group_reduce_enabled}
                    onChange={(e) => onChange({ ...value, summary_group_reduce_enabled: e.target.checked })}
                />
                summary_group_reduce_enabled
            </label>
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
    return (
        <div className="rule-box">
            {rules.map((rule, idx) => (
                <div key={idx} className="rule-item">
                    <div className="row between">
                        <strong>Rule {idx + 1}</strong>
                        <button
                            onClick={() => {
                                const next = [...rules];
                                next.splice(idx, 1);
                                onChange(next);
                            }}
                        >
                            删除 Rule
                        </button>
                    </div>
                    <div className="grid2">
                        <label className="row gap-8">
                            <input
                                type="checkbox"
                                checked={rule.enabled}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, enabled: e.target.checked };
                                    onChange(next);
                                }}
                            />
                            enabled
                        </label>
                        <div>
                            <label>chat_type</label>
                            <select
                                value={rule.chat_type}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, chat_type: e.target.value as 'group' | 'private' };
                                    onChange(next);
                                }}
                            >
                                <option value="group">group</option>
                                <option value="private">private</option>
                            </select>
                        </div>
                        <div>
                            <label>number</label>
                            <input
                                value={rule.number}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, number: e.target.value };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div>
                            <label>trigger_mode</label>
                            <select
                                value={rule.trigger_mode}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, trigger_mode: e.target.value };
                                    onChange(next);
                                }}
                            >
                                <option value="always">always</option>
                                <option value="keyword">keyword</option>
                                <option value="at_bot">at_bot</option>
                                <option value="ai_decide">ai_decide</option>
                                <option value="ai_decide || keyword">ai_decide || keyword</option>
                                <option value="at_bot || keyword">at_bot || keyword</option>
                            </select>
                        </div>
                        <div>
                            <label>temperature (可选)</label>
                            <input
                                type="number"
                                step="0.1"
                                value={rule.temperature ?? 0.4}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, temperature: Number(e.target.value) };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div className="full-row">
                            <label>keywords</label>
                            <StringListEditor
                                values={rule.keywords}
                                onChange={(keywords) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, keywords };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div className="full-row">
                            <label>ai_decision_prompt</label>
                            <textarea
                                rows={6}
                                value={rule.ai_decision_prompt}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, ai_decision_prompt: e.target.value };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div className="full-row">
                            <label>reply_prompt</label>
                            <textarea
                                rows={6}
                                value={rule.reply_prompt}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, reply_prompt: e.target.value };
                                    onChange(next);
                                }}
                            />
                        </div>
                    </div>
                </div>
            ))}
            <button onClick={() => onChange([...rules, makeDefaultAutoRule()])}>+ 添加 Rule</button>
        </div>
    );
}

function DidaRulesEditor({ rules, onChange }: { rules: DidaRule[]; onChange: (next: DidaRule[]) => void }) {
    return (
        <div className="rule-box">
            {rules.map((rule, idx) => (
                <div key={idx} className="rule-item">
                    <div className="row between">
                        <strong>Rule {idx + 1}</strong>
                        <button
                            onClick={() => {
                                const next = [...rules];
                                next.splice(idx, 1);
                                onChange(next);
                            }}
                        >
                            删除 Rule
                        </button>
                    </div>
                    <div className="grid2">
                        <label className="row gap-8">
                            <input
                                type="checkbox"
                                checked={rule.enabled}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, enabled: e.target.checked };
                                    onChange(next);
                                }}
                            />
                            enabled
                        </label>
                        <label className="row gap-8">
                            <input
                                type="checkbox"
                                checked={rule.dida_enabled}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, dida_enabled: e.target.checked };
                                    onChange(next);
                                }}
                            />
                            dida_enabled
                        </label>
                        <div>
                            <label>chat_type</label>
                            <select
                                value={rule.chat_type}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, chat_type: e.target.value as 'group' | 'private' };
                                    onChange(next);
                                }}
                            >
                                <option value="group">group</option>
                                <option value="private">private</option>
                            </select>
                        </div>
                        <div>
                            <label>number</label>
                            <input
                                value={rule.number}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, number: e.target.value };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div>
                            <label>trigger_mode</label>
                            <select
                                value={rule.trigger_mode}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, trigger_mode: e.target.value };
                                    onChange(next);
                                }}
                            >
                                <option value="ai_decide">ai_decide</option>
                                <option value="always">always</option>
                                <option value="keyword">keyword</option>
                            </select>
                        </div>
                        <div>
                            <label>action_temperature</label>
                            <input
                                type="number"
                                step="0.1"
                                value={rule.action_temperature}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, action_temperature: Number(e.target.value) };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div>
                            <label>reply_temperature</label>
                            <input
                                type="number"
                                step="0.1"
                                value={rule.reply_temperature}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, reply_temperature: Number(e.target.value) };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div className="full-row">
                            <label>ai_decision_prompt</label>
                            <textarea
                                rows={6}
                                value={rule.ai_decision_prompt}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, ai_decision_prompt: e.target.value };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div className="full-row">
                            <label>action_prompt</label>
                            <textarea
                                rows={6}
                                value={rule.action_prompt}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, action_prompt: e.target.value };
                                    onChange(next);
                                }}
                            />
                        </div>
                        <div className="full-row">
                            <label>reply_prompt</label>
                            <textarea
                                rows={6}
                                value={rule.reply_prompt}
                                onChange={(e) => {
                                    const next = [...rules];
                                    next[idx] = { ...rule, reply_prompt: e.target.value };
                                    onChange(next);
                                }}
                            />
                        </div>
                    </div>
                </div>
            ))}
            <button onClick={() => onChange([...rules, makeDefaultDidaRule()])}>+ 添加 Rule</button>
        </div>
    );
}

function AutoReplyForm({ value, onChange }: { value: AutoReplyConfig; onChange: (next: AutoReplyConfig) => void }) {
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
            <div>
                <label>context_history_limit</label>
                <input
                    type="number"
                    value={value.context_history_limit}
                    onChange={(e) => onChange({ ...value, context_history_limit: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>context_max_chars</label>
                <input
                    type="number"
                    value={value.context_max_chars}
                    onChange={(e) => onChange({ ...value, context_max_chars: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>context_window_seconds</label>
                <input
                    type="number"
                    value={value.context_window_seconds}
                    onChange={(e) => onChange({ ...value, context_window_seconds: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>min_reply_interval_seconds</label>
                <input
                    type="number"
                    value={value.min_reply_interval_seconds}
                    onChange={(e) => onChange({ ...value, min_reply_interval_seconds: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>flush_check_interval_seconds</label>
                <input
                    type="number"
                    value={value.flush_check_interval_seconds}
                    onChange={(e) => onChange({ ...value, flush_check_interval_seconds: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>pending_expire_seconds</label>
                <input
                    type="number"
                    value={value.pending_expire_seconds}
                    onChange={(e) => onChange({ ...value, pending_expire_seconds: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>pending_max_messages</label>
                <input
                    type="number"
                    value={value.pending_max_messages}
                    onChange={(e) => onChange({ ...value, pending_max_messages: Number(e.target.value) })}
                />
            </div>
            <label className="row gap-8">
                <input
                    type="checkbox"
                    checked={value.bypass_cooldown_when_at_bot}
                    onChange={(e) => onChange({ ...value, bypass_cooldown_when_at_bot: e.target.checked })}
                />
                bypass_cooldown_when_at_bot
            </label>
            <div className="full-row">
                <label>rules</label>
                <AutoReplyRulesEditor rules={value.rules} onChange={(rules) => onChange({ ...value, rules })} />
            </div>
        </div>
    );
}

function DidaForm({ value, onChange }: { value: DidaConfig; onChange: (next: DidaConfig) => void }) {
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
            <div>
                <label>bot_qq</label>
                <input value={value.bot_qq} onChange={(e) => onChange({ ...value, bot_qq: e.target.value })} />
            </div>
            <div className="full-row">
                <label>admin_qqs</label>
                <StringListEditor values={value.admin_qqs} onChange={(admin_qqs) => onChange({ ...value, admin_qqs })} />
            </div>
            <div>
                <label>context_history_limit</label>
                <input
                    type="number"
                    value={value.context_history_limit}
                    onChange={(e) => onChange({ ...value, context_history_limit: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>context_max_chars</label>
                <input
                    type="number"
                    value={value.context_max_chars}
                    onChange={(e) => onChange({ ...value, context_max_chars: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>context_window_seconds</label>
                <input
                    type="number"
                    value={value.context_window_seconds}
                    onChange={(e) => onChange({ ...value, context_window_seconds: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>min_reply_interval_seconds</label>
                <input
                    type="number"
                    value={value.min_reply_interval_seconds}
                    onChange={(e) => onChange({ ...value, min_reply_interval_seconds: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>flush_check_interval_seconds</label>
                <input
                    type="number"
                    value={value.flush_check_interval_seconds}
                    onChange={(e) => onChange({ ...value, flush_check_interval_seconds: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>pending_expire_seconds</label>
                <input
                    type="number"
                    value={value.pending_expire_seconds}
                    onChange={(e) => onChange({ ...value, pending_expire_seconds: Number(e.target.value) })}
                />
            </div>
            <div>
                <label>pending_max_messages</label>
                <input
                    type="number"
                    value={value.pending_max_messages}
                    onChange={(e) => onChange({ ...value, pending_max_messages: Number(e.target.value) })}
                />
            </div>
            <label className="row gap-8">
                <input
                    type="checkbox"
                    checked={value.bypass_cooldown_when_at_bot}
                    onChange={(e) => onChange({ ...value, bypass_cooldown_when_at_bot: e.target.checked })}
                />
                bypass_cooldown_when_at_bot
            </label>
            <div className="full-row">
                <label>rules</label>
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

                        return (
                            <div key={field.key}>
                                <label>{field.label}</label>
                                <input
                                    type={field.type === 'number' ? 'number' : 'text'}
                                    value={
                                        field.type === 'number'
                                            ? String(toNumber(params[field.key], 0))
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
        <div className="rule-box" style={{ marginLeft: depth * 12 }}>
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
                                            params: node.params || {},
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
                                params={node.params || {}}
                                onChange={(nextParams) => updateAt(idx, { ...node, params: nextParams })}
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

export default function App() {
    const [tab, setTab] = useState<TabKey>('basic');
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
        restart_policy: 'docker-compose',
    });
    const [deployLogs, setDeployLogs] = useState<string[]>([]);
    const [snapshots, setSnapshots] = useState<string[]>([]);
    const [selectedScheduleIndex, setSelectedScheduleIndex] = useState(0);
    const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
    const [timelineCount, setTimelineCount] = useState(20);
    const [schedulerMessage, setSchedulerMessage] = useState('');

    const fixedSections = useMemo(() => normalizeFixedSections(agent), [agent]);
    const schedulerConfig = useMemo(() => getSchedulerManagerConfig(agent), [agent]);

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
        model: toStringValue(fixedSections.dida_agent_config.config.model),
        decision_model: toStringValue(fixedSections.dida_agent_config.config.decision_model),
        temperature: toNumber(fixedSections.dida_agent_config.config.temperature, 0.1),
        bot_qq: toStringValue(fixedSections.dida_agent_config.config.bot_qq),
        admin_qqs: toStringArray(fixedSections.dida_agent_config.config.admin_qqs),
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

    const selectedSchedule = schedulerConfig.schedules[selectedScheduleIndex];

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
                    <div className="row between">
                        <h2>固定 Agent 表单</h2>
                        <button onClick={saveAgentFull}>保存 Agent 配置</button>
                    </div>
                    <div>
                        <label>选择 Agent 类型</label>
                        <select
                            value={selectedAgent}
                            onChange={(e) => setSelectedAgent(e.target.value as FixedAgentKey)}
                        >
                            {FIXED_AGENT_OPTIONS.map((option) => (
                                <option key={option.key} value={option.key}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
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
                            onChange={(next) => updateFixedSectionConfig('dida_agent_config', next as unknown as JsonObject)}
                        />
                    )}
                </section>
            )}

            {tab === 'scheduler' && (
                <section className="panel">
                    <h2>Scheduler 设置</h2>
                    <details className="scheduler-help top-gap" open>
                        <summary>Scheduler 功能使用说明</summary>
                        <div className="scheduler-help-content">
                            <h4>1. 可用参数</h4>
                            <ul>
                                <li>timezone: 调度时区，例如 Asia/Shanghai。cron 计算会使用该时区。</li>
                                <li>timeline count: 时间线预览条数，上限建议不超过 200。</li>
                                <li>name: 任务名称，仅用于识别和展示。</li>
                                <li>type: 调度类型，可选 cron 或 interval。</li>
                                <li>expression: cron 表达式，仅在 type=cron 时生效。</li>
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
                                <li>先创建 schedule 并设置 type、expression/seconds、enabled。</li>
                                <li>在 steps_tree 里先搭结构（group），再填 action 参数。</li>
                                <li>点击刷新时间线确认触发节奏。</li>
                                <li>点击保存并编译 Scheduler 落盘生效。</li>
                            </ol>
                        </div>
                    </details>

                    <div className="row gap-8 wrap">
                        <div className="grow">
                            <label>timezone</label>
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
                            <label>timeline count</label>
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
                                        <span>
                                            {item.name || `schedule_${idx + 1}`} ({item.type}) {item.enabled ? '启用' : '停用'}
                                        </span>
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

                    {selectedSchedule ? (
                        <div className="card top-gap">
                            <h3>编辑 Schedule</h3>
                            <div className="grid2">
                                <div>
                                    <label>name</label>
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
                                    <label>type</label>
                                    <select
                                        value={selectedSchedule.type}
                                        onChange={(e) => {
                                            const nextType = e.target.value as 'cron' | 'interval';
                                            const nextSchedule: ScheduleConfig = {
                                                ...selectedSchedule,
                                                type: nextType,
                                            };
                                            if (nextType === 'cron') {
                                                nextSchedule.expression = selectedSchedule.expression || '*/5 * * * *';
                                                delete nextSchedule.seconds;
                                            } else {
                                                nextSchedule.seconds = selectedSchedule.seconds || 60;
                                                delete nextSchedule.expression;
                                            }
                                            updateScheduleAt(selectedScheduleIndex, nextSchedule);
                                        }}
                                    >
                                        <option value="cron">cron</option>
                                        <option value="interval">interval</option>
                                    </select>
                                </div>
                                <label className="row gap-8">
                                    <input
                                        type="checkbox"
                                        checked={selectedSchedule.enabled}
                                        onChange={(e) =>
                                            updateScheduleAt(selectedScheduleIndex, {
                                                ...selectedSchedule,
                                                enabled: e.target.checked,
                                            })
                                        }
                                    />
                                    enabled
                                </label>
                                {selectedSchedule.type === 'cron' ? (
                                    <div>
                                        <label>expression</label>
                                        <input
                                            value={selectedSchedule.expression || ''}
                                            onChange={(e) =>
                                                updateScheduleAt(selectedScheduleIndex, {
                                                    ...selectedSchedule,
                                                    expression: e.target.value,
                                                })
                                            }
                                        />
                                    </div>
                                ) : (
                                    <div>
                                        <label>seconds</label>
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

                            <div className="top-gap">
                                <label>steps_tree</label>
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
                        </div>
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
                </section>
            )}
        </div>
    );
}
