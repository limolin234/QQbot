import React, { useEffect, useRef, useState } from 'react';

const BASE = (import.meta.env.VITE_CONFIG_STUDIO_API_BASE as string | undefined)?.trim()?.replace(/\/$/, '') || '';

export function TerminalPanel({ deployConfig }: { deployConfig: any }) {
    const [command, setCommand] = useState('');
    const [output, setOutput] = useState<string[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const outputRef = useRef<HTMLDivElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    // Auto scroll
    useEffect(() => {
        if (outputRef.current) {
            outputRef.current.scrollTop = outputRef.current.scrollHeight;
        }
    }, [output]);

    const presets = [
        { label: '查看 Docker 状态', cmd: 'docker ps' },
        { label: '查看系统信息', cmd: 'uname -a && uptime' },
        { label: 'napcat 日志', cmd: 'docker logs -f --tail 50 napcat' },
        { label: 'qqbot 日志', cmd: 'docker logs -f --tail 50 qqbot' },

    ];

    const runCommand = async (cmdStr: string) => {
        if (!cmdStr.trim()) return;

        // Clear previous connection
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        setOutput([`> [SSH to ${deployConfig?.host || 'remote'}] $ ${cmdStr}`]);
        setIsRunning(true);
        setCommand(cmdStr);

        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        const targetUrl = `${BASE}/api/sys/command_ssh`;

        try {
            const response = await fetch(targetUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...deployConfig, cmd: cmdStr }),
                signal: abortController.signal
            });

            if (!response.body) throw new Error("No response body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                if (lines.length > 0) {
                    setOutput((prev) => {
                        const newArr = [...prev, ...lines];
                        if (newArr.length > 2000) return newArr.slice(-2000);
                        return newArr;
                    });
                }
            }

            if (buffer) {
                setOutput(prev => [...prev, buffer]);
            }

            setIsRunning(false);
        } catch (err: any) {
            if (err.name === 'AbortError') {
                setOutput((prev) => [...prev, '\n--- 已手动中止 ---']);
            } else {
                console.error('SSH Error:', err);
                setOutput((prev) => [...prev, `\n--- 请求出错: ${err.message || err} ---`]);
            }
            setIsRunning(false);
        }
    };

    const handleStop = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        setIsRunning(false);
    };

    return (
        <div style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "16px", minHeight: "100vh", boxSizing: "border-box", background: "#f5f5f5" }}>
            <h2 style={{ margin: 0, color: "#333", fontSize: "20px", fontWeight: "bold" }}>SSH Web Terminal ({deployConfig?.host || "未配置服务器 IP"})</h2>
            
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center", background: "#fff", padding: "12px", borderRadius: "8px", boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}>
                <span style={{ color: "#666", fontWeight: "bold", marginRight: "8px" }}>快捷指令:</span>
                {presets.map(p => (
                    <button 
                        key={p.cmd} 
                        style={{ cursor: "pointer", padding: "6px 12px", background: "#e6f7ff", color: "#1890ff", border: "1px solid #91d5ff", borderRadius: "4px", fontSize: "13px", whiteSpace: "nowrap", flex: "0 0 auto", margin: 0, width: "max-content", display: "inline-block" }}
                        onClick={() => runCommand(p.cmd)}
                        disabled={isRunning}
                    >
                        {p.label}
                    </button>
                ))}
            </div>

            <div style={{ display: "flex", gap: "12px", background: "#fff", padding: "16px", borderRadius: "8px", boxShadow: "0 1px 3px rgba(0,0,0,0.1)", alignItems: "center" }}>
                <input 
                    style={{ flex: 1, padding: "12px", border: "2px solid #d9d9d9", borderRadius: "6px", outline: "none", fontSize: "14px", minWidth: "100px", background: "#fff", color: "#333", margin: 0 }}
                    placeholder="在此输入自定义 bash命令 (例如: ls -lh /)..." 
                    value={command}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCommand(e.target.value)}
                    onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                        if (e.key === "Enter") runCommand(command);
                    }}
                    disabled={isRunning}
                />
                {!isRunning ? (
                    <button 
                        style={{ padding: "8px 16px", background: "#1890ff", color: "white", border: "none", borderRadius: "6px", cursor: "pointer", fontWeight: "bold", fontSize: "14px", flexShrink: 0, margin: 0, width: "max-content", minWidth: "60px", display: "inline-block", height: "auto" }}
                        onClick={() => runCommand(command)}
                    >
                        执行
                    </button>
                ) : (
                    <button 
                        style={{ padding: "8px 16px", background: "#ff4d4f", color: "white", border: "none", borderRadius: "6px", cursor: "pointer", fontWeight: "bold", fontSize: "14px", flexShrink: 0, margin: 0, width: "max-content", minWidth: "60px", display: "inline-block", height: "auto" }}
                        onClick={handleStop}
                    >
                        停止
                    </button>
                )}
            </div>

            <div 
                ref={outputRef}
                style={{
                    backgroundColor: "#1e1e1e",
                    color: "#00ff00",
                    padding: "20px",
                    borderRadius: "8px",
                    flex: 1,
                    overflowY: "auto",
                    fontFamily: "Menlo, Monaco, Consolas, 'Courier New', monospace",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-all",
                    fontSize: "13px",
                    lineHeight: 1.6,
                    minHeight: "500px",
                    boxShadow: "inset 0 2px 4px rgba(0,0,0,0.5)",
                    border: "1px solid #000"
                }}
            >
                {output.length === 0 ? (
                    <span style={{ color: "#888" }}>命令行已就绪。请从上方选择快捷指令或手动输入命令。</span>
                ) : (
                    output.map((line, idx) => (
                        <div key={idx} style={{ minHeight: "20px" }}>{line}</div>
                    ))
                )}
            </div>
            
            <div style={{ marginTop: "auto" }}>
                <span style={{ fontSize: "13px", color: "#999", display: "block", textAlign: "center", padding: "10px 0" }}>
                    💡 提示：某些持续输出的命令（如 docker logs -f 或者 ping）可以点击停止按钮中断。
                </span>
            </div>
        </div>
    );
}

