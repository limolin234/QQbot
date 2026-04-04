import type { AllConfigResponse, ScheduleConfig, TimelineEvent } from './types';

const RAW_BASE = (import.meta.env.VITE_CONFIG_STUDIO_API_BASE as string | undefined)?.trim() || '';
const BASE = RAW_BASE.replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const url = `${BASE}${path}`;
    let response: Response;
    try {
        response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...(init?.headers || {}),
            },
            ...init,
        });
    } catch (error) {
        throw new Error(
            `Network error: Failed to fetch ${url}. Please ensure backend API is running and reachable. Original: ${String(error)}`,
        );
    }

    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed: ${response.status}`);
    }

    return response.json() as Promise<T>;
}

export function getAllConfig() {
    return request<AllConfigResponse>('/api/config/all');
}

export function saveEnv(llmApiKey: string, llmApiBaseUrl: string) {
    return request('/api/config/env', {
        method: 'PUT',
        body: JSON.stringify({ llm_api_key: llmApiKey, llm_api_base_url: llmApiBaseUrl }),
    });
}

export function saveAgent(data: Record<string, unknown>) {
    return request('/api/config/agent', {
        method: 'PUT',
        body: JSON.stringify({ data }),
    });
}

export function saveAgentRaw(rawYaml: string) {
    return request('/api/config/agent/raw', {
        method: 'PUT',
        body: JSON.stringify({ raw_yaml: rawYaml }),
    });
}

export function getAgentRaw() {
    return request<{ raw_yaml: string }>('/api/config/agent/raw');
}

export function compileSchedules(schedules: ScheduleConfig[]) {
    return request<{ schedules: ScheduleConfig[] }>('/api/scheduler/compile', {
        method: 'POST',
        body: JSON.stringify({ schedules }),
    });
}

export function getTimeline(timezone: string, schedules: ScheduleConfig[], count = 10) {
    return request<{ events: TimelineEvent[] }>('/api/scheduler/timeline', {
        method: 'POST',
        body: JSON.stringify({ timezone, schedules, count }),
    });
}

export function testConnection(payload: Record<string, unknown>) {
    return request<{ success: boolean; message: string; logs: string[] }>('/api/deploy/test-connection', {
        method: 'POST',
        body: JSON.stringify(payload),
    });
}

export function deployPush(payload: Record<string, unknown>) {
    return request<{ success: boolean; message: string; logs: string[] }>('/api/deploy/push', {
        method: 'POST',
        body: JSON.stringify(payload),
    });
}

export function deployPull(payload: Record<string, unknown>) {
    return request<{ success: boolean; message: string; logs: string[] }>('/api/deploy/pull', {
        method: 'POST',
        body: JSON.stringify(payload),
    });
}

export function listSnapshots() {
    return request<{ snapshots: string[] }>('/api/history');
}

export function restoreSnapshot(snapshotName: string, target: 'env' | 'agent') {
    return request('/api/history/restore', {
        method: 'POST',
        body: JSON.stringify({ snapshot_name: snapshotName, target }),
    });
}
