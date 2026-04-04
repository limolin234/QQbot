export type TabKey = 'basic' | 'agent' | 'scheduler' | 'deploy' | 'history';

export type StepNode = {
    id: string;
    kind: 'action' | 'group' | 'if';
    action?: string;
    params?: Record<string, unknown>;
    name?: string;
    children?: StepNode[];
    condition?: {
        source?: string;
        key?: string;
        op?: string;
        value?: string;
    };
    then_steps?: StepNode[];
    else_steps?: StepNode[];
};

export type ScheduleConfig = {
    name: string;
    type: 'cron' | 'interval';
    expression?: string;
    seconds?: number;
    enabled: boolean;
    steps?: Array<{ action: string; params: Record<string, unknown> }>;
    steps_tree?: StepNode[];
};

export type AgentConfigRoot = {
    scheduler_manager?: {
        file_name: string;
        config: {
            timezone?: string;
            schedules?: ScheduleConfig[];
        };
    };
    [key: string]: unknown;
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
