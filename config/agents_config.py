"""Agent 配置文件 - 定义所有 Agent 的配置"""

# Agent 配置字典
AGENTS_CONFIG = {
    "simple_chat": {
        "enabled": True,
        "name": "聊天助手",
        "description": "友好的群聊助手，回答问题和闲聊",
        "class": "SimpleChatAgent",
        "config": {
            "model": "deepseek-v3",
            "system_prompt": """你是一个友好、热情的 QQ 群助手。你的职责是：
1. 回答群友的问题
2. 进行友好的闲聊
3. 提供帮助和建议
4. 保持积极正面的态度

注意：
- 回复要简洁明了，不要过长
- 使用轻松友好的语气
- 适当使用表情符号
- 不要重复用户的问题""",
            "monitored_groups": [1075786046],  # 监听的群号列表
            "trigger_mode": "hybrid",  # 触发模式：all（全部监听）/ keywords（关键词）/ hybrid（混合）
            "wake_words": [r"小助手", r"bot", r"机器人", r"助手"],  # 唤醒关键词
            "require_at": False,  # 是否必须 @ 才响应
        }
    },

    "notification": {
        "enabled": True,
        "name": "通知摘要助手",
        "description": "识别重要通知并发送私聊摘要",
        "class": "NotificationAgent",
        "config": {
            "model": "deepseek-v3",
            "monitored_groups": [1075786046],  # 通知群号（需要修改为实际的通知群号）
            "target_user": 3291053545,  # 接收摘要的用户 QQ 号（需要修改为实际的 QQ 号）
            "trigger_mode": "all",  # 通知群全部监听（消息频率不高）
            "keywords": [],  # 关键词列表（trigger_mode=keywords 或 hybrid 时使用）
            "notification_prompt": """你是一个通知识别助手。判断以下消息是否为重要通知。

重要通知包括：
- 作业通知（布置作业、作业截止日期）
- 考试安排（考试时间、地点、科目）
- 课程变更（调课、停课、补课）
- 重要活动通知（讲座、会议、活动）
- 截止日期提醒（报名、提交材料等）
- 成绩公布
- 学校通知

不重要的消息：
- 日常闲聊
- 问候语
- 无关紧要的讨论

请分析以下消息，返回 JSON 格式：
- 如果是重要通知：{"is_important": true, "category": "通知类别", "summary": "简洁的摘要（50字以内）", "key_info": "关键信息（时间、地点等）"}
- 如果不是：{"is_important": false}

消息内容：{message}""",
            "summary_prompt": """请为以下重要通知生成一个清晰、简洁的摘要，包含所有关键信息。

原始消息：{message}

请生成摘要，格式如下：
📢 【通知类别】
📝 内容：简要描述
⏰ 时间：（如果有）
📍 地点：（如果有）
⚠️ 注意事项：（如果有）"""
        }
    }
}

# Supervisor 配置
SUPERVISOR_CONFIG = {
    "model": "deepseek-v3",
    "system_prompt": """你是一个 Agent 路由器。根据消息内容和上下文，决定应该调用哪个 Agent。

可用的 Agent：
1. simple_chat - 聊天助手：处理日常对话、问答、闲聊
2. notification - 通知摘要助手：识别重要通知并发送摘要

规则：
- 如果消息来自通知群（禁言群），优先使用 notification
- 如果消息包含 @ 或唤醒词，使用 simple_chat
- 如果消息看起来是通知（作业、考试、活动等），使用 notification
- 可以同时触发多个 Agent（返回多个 Agent ID）

请分析消息，返回 JSON 格式：
{"agents": ["agent_id1", "agent_id2"], "reason": "选择原因"}

如果不需要任何 Agent 处理，返回：
{"agents": [], "reason": "原因"}"""
}

# CLI 面板配置
CLI_PANEL_CONFIG = {
    "enabled": True,          # 是否启用 CLI
    "mode": "interactive",    # 模式：interactive（交互式）/ panel（仅面板）/ command（仅命令）
    "refresh_rate": 1,        # 刷新频率（秒），仅 panel 和 interactive 模式有效
    "show_stats": True,       # 是否显示统计信息
}
