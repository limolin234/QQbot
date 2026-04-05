export type TabKey = 'basic' | 'agent' | 'scheduler' | 'deploy' | 'history' | 'terminal';

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

export type StepNode = {
    id: string;
    kind: 'action' | 'group';
    action?: string;
    params?: JsonObject;
    name?: string;
    children?: StepNode[];
};

export type ScheduleConfig = {
    name: string;
    type: 'cron' | 'interval';
    expression?: string;
    seconds?: number;
    enabled: boolean;
    steps?: Array<{ action: string; params: JsonObject }>;
    steps_tree?: StepNode[];
};

export type AgentConfigRoot = {
    [key: string]: JsonValue;
};

export type AllConfigResponse = {
    env: {
        LLM_API_KEY: string;
        LLM_API_BASE_URL: string;
    };
    agent: AgentConfigRoot;
};

export type TimelineEvent = {
    schedule_name: string;
    trigger_at: string;
    source: 'cron' | 'interval';
};
