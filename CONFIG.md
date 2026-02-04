# 配置说明文档

本文档详细说明 QQ Bot 多 Agent 系统的所有配置选项。

## 📁 配置文件概览

| 文件 | 用途 | 说明 |
|------|------|------|
| `.env` | 环境变量 | API Key 等敏感信息 |
| `config/napcat_config.py` | NapCat 连接 | WebSocket 地址配置 |
| `config/bot_config.py` | Bot 基础配置 | 模型、安全限制等 |
| `config/agents_config.py` | Agent 配置 | 所有 Agent 的配置 |

---

## 1. 环境变量配置 (`.env`)

### 必需配置

```env
# LLM API 配置
YUNWU_API_KEY=sk-xxx...
API_BASE_URL=https://yunwu.ai/v1
```

### 配置说明

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `YUNWU_API_KEY` | OpenAI 兼容 API 的密钥 | `sk-xxx...` |
| `API_BASE_URL` | API 基础 URL | `https://yunwu.ai/v1` |

### 支持的 API 提供商

- **Yunwu AI**: `https://yunwu.ai/v1`
- **DeepSeek**: `https://api.deepseek.com/v1`
- **OpenAI**: `https://api.openai.com/v1`
- 其他 OpenAI 兼容 API

---

## 2. NapCat 连接配置 (`config/napcat_config.py`)

```python
# WebSocket 服务端地址
NAPCAT_WS_URL = "ws://127.0.0.1:3001"

# HTTP API 地址（可选，暂未使用）
NAPCAT_HTTP_URL = "http://127.0.0.1:3000"
```

### 配置说明

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `NAPCAT_WS_URL` | WebSocket 服务端地址 | `ws://127.0.0.1:3001` |
| `NAPCAT_HTTP_URL` | HTTP API 地址 | `http://127.0.0.1:3000` |

**注意**：
- 端口 `3001` 是 OneBot 协议端口，不是 WebUI 端口（6099）
- 本地连接无需 Token 认证

---

## 3. Bot 基础配置 (`config/bot_config.py`)

```python
# 监听的群号列表
MONITORED_GROUPS = [1075786046]

# Bot QQ 号（自动获取，无需手动配置）
BOT_QQ = None

# LLM 模型 ID
MODEL_ID = "deepseek-v3"

# Agent 系统提示词
AGENT_SYSTEM_PROMPT = """你是一个友好、乐于助人的 QQ 群助手。

你的特点：
- 回答简洁明了，不啰嗦
- 语气轻松友好，适合群聊氛围
- 能够记住对话历史，提供连贯的对话体验
- 遇到不确定的问题会诚实地说不知道

请用中文回复，保持友好和专业。"""

# 唤醒关键词（支持正则表达式）
WAKE_WORDS = [
    r"小助手",
    r"bot",
    r"机器人",
    r"助手"
]

# 安全配置
MAX_MESSAGE_LENGTH = 500          # 最大消息长度（字符）
MAX_MESSAGES_PER_MINUTE = 10      # 每分钟最大消息数
VIOLATION_THRESHOLD = 3           # 违规阈值（超过后暂停响应）
```

### 配置说明

#### 基础配置

| 变量名 | 说明 | 类型 | 默认值 |
|--------|------|------|--------|
| `MONITORED_GROUPS` | 监听的群号列表 | `List[int]` | `[1075786046]` |
| `BOT_QQ` | Bot QQ 号 | `int` | 自动获取 |

#### Agent 配置

| 变量名 | 说明 | 类型 | 默认值 |
|--------|------|------|--------|
| `MODEL_ID` | LLM 模型 ID | `str` | `"deepseek-v3"` |
| `AGENT_SYSTEM_PROMPT` | 系统提示词 | `str` | 见上方 |

#### 唤醒配置

| 变量名 | 说明 | 类型 | 默认值 |
|--------|------|------|--------|
| `WAKE_WORDS` | 唤醒关键词列表 | `List[str]` | 见上方 |

**注意**：关键词支持正则表达式，如 `r"小助手"` 会匹配包含"小助手"的消息。

#### 安全配置

| 变量名 | 说明 | 类型 | 默认值 |
|--------|------|------|--------|
| `MAX_MESSAGE_LENGTH` | 最大消息长度 | `int` | `500` |
| `MAX_MESSAGES_PER_MINUTE` | 每分钟最大消息数 | `int` | `10` |
| `VIOLATION_THRESHOLD` | 违规阈值 | `int` | `3` |

---

## 4. Agent 配置 (`config/agents_config.py`)

### 4.1 Agent 配置结构

```python
AGENTS_CONFIG = {
    "agent_id": {
        "enabled": True,              # 是否启用
        "name": "Agent 名称",         # 显示名称
        "description": "Agent 描述",  # 功能描述
        "class": "AgentClassName",    # Agent 类名
        "config": {
            # Agent 特定配置
        }
    }
}
```

### 4.2 SimpleChatAgent 配置

```python
"simple_chat": {
    "enabled": True,
    "name": "聊天助手",
    "description": "友好的群聊助手，回答问题和闲聊",
    "class": "SimpleChatAgent",
    "config": {
        "model": "deepseek-v3",
        "system_prompt": "你是一个友好的群助手...",
        "monitored_groups": [1075786046],
        "trigger_mode": "hybrid",
        "wake_words": [r"小助手", r"bot", r"机器人"],
        "require_at": False
    }
}
```

#### 配置项说明

| 配置项 | 说明 | 类型 | 默认值 | 必需 |
|--------|------|------|--------|------|
| `model` | LLM 模型 ID | `str` | `"deepseek-v3"` | ✅ |
| `system_prompt` | 系统提示词 | `str` | - | ✅ |
| `monitored_groups` | 监听的群号列表 | `List[int]` | `[]` | ✅ |
| `trigger_mode` | 触发模式 | `str` | `"hybrid"` | ✅ |
| `wake_words` | 唤醒关键词 | `List[str]` | `[]` | ❌ |
| `require_at` | 是否必须 @ | `bool` | `False` | ❌ |

#### 触发模式说明

| 模式 | 说明 | API 消耗 | 适用场景 |
|------|------|---------|---------|
| `all` | 监听所有消息 | 高 | 低频群 |
| `keywords` | 仅关键词触发 | 低 | 节省成本 |
| `hybrid` | 关键词 + LLM 分析 | 中 | 聊天群（推荐） |

### 4.3 NotificationAgent 配置

```python
"notification": {
    "enabled": True,
    "name": "通知摘要助手",
    "description": "识别重要通知并发送私聊摘要",
    "class": "NotificationAgent",
    "config": {
        "model": "deepseek-v3",
        "monitored_groups": [123456789],
        "target_user": 987654321,
        "trigger_mode": "all",
        "keywords": [],
        "notification_prompt": "...",
        "summary_prompt": "..."
    }
}
```

#### 配置项说明

| 配置项 | 说明 | 类型 | 默认值 | 必需 |
|--------|------|------|--------|------|
| `model` | LLM 模型 ID | `str` | `"deepseek-v3"` | ✅ |
| `monitored_groups` | 监听的群号列表 | `List[int]` | `[]` | ✅ |
| `target_user` | 接收摘要的 QQ 号 | `int` | - | ✅ |
| `trigger_mode` | 触发模式 | `str` | `"all"` | ✅ |
| `keywords` | 关键词列表 | `List[str]` | `[]` | ❌ |
| `notification_prompt` | 通知识别提示词 | `str` | - | ✅ |
| `summary_prompt` | 摘要生成提示词 | `str` | - | ❌ |

#### 通知识别提示词示例

```python
notification_prompt = """你是一个通知识别助手。判断以下消息是否为重要通知。

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

消息内容：{message}"""
```

### 4.4 CLI 面板配置

```python
CLI_PANEL_CONFIG = {
    "enabled": True,      # 是否启用 CLI 面板
    "refresh_rate": 1,    # 刷新频率（秒）
    "show_stats": True    # 是否显示统计信息
}
```

#### 配置项说明

| 配置项 | 说明 | 类型 | 默认值 |
|--------|------|------|--------|
| `enabled` | 是否启用 CLI 面板 | `bool` | `True` |
| `refresh_rate` | 刷新频率（秒） | `float` | `1` |
| `show_stats` | 是否显示统计信息 | `bool` | `True` |

---

## 5. 配置示例

### 5.1 单 Agent 配置（仅聊天助手）

```python
AGENTS_CONFIG = {
    "simple_chat": {
        "enabled": True,
        "name": "聊天助手",
        "class": "SimpleChatAgent",
        "config": {
            "model": "deepseek-v3",
            "monitored_groups": [1075786046],
            "trigger_mode": "hybrid",
            "wake_words": [r"小助手", r"bot"],
            "require_at": False,
            "system_prompt": "你是一个友好的群助手..."
        }
    }
}
```

### 5.2 多 Agent 配置（聊天 + 通知）

```python
AGENTS_CONFIG = {
    "simple_chat": {
        "enabled": True,
        "name": "聊天助手",
        "class": "SimpleChatAgent",
        "config": {
            "model": "deepseek-v3",
            "monitored_groups": [1075786046],  # 聊天群
            "trigger_mode": "hybrid",
            "wake_words": [r"小助手", r"bot"],
            "require_at": False,
            "system_prompt": "你是一个友好的群助手..."
        }
    },
    "notification": {
        "enabled": True,
        "name": "通知摘要助手",
        "class": "NotificationAgent",
        "config": {
            "model": "deepseek-v3",
            "monitored_groups": [123456789],  # 通知群
            "target_user": 987654321,  # 你的大号 QQ
            "trigger_mode": "all",
            "notification_prompt": "..."
        }
    }
}
```

### 5.3 禁用 Agent

```python
"notification": {
    "enabled": False,  # 禁用此 Agent
    # ... 其他配置
}
```

---

## 6. 常见配置场景

### 场景 1：只在特定群响应

```python
"monitored_groups": [1075786046, 123456789]
```

### 场景 2：必须 @ 才响应

```python
"require_at": True
```

### 场景 3：节省 API 调用（仅关键词触发）

```python
"trigger_mode": "keywords",
"wake_words": [r"小助手", r"bot", r"帮助"]
```

### 场景 4：多个通知群

```python
"notification": {
    "config": {
        "monitored_groups": [111111, 222222, 333333],
        "target_user": 987654321
    }
}
```

### 场景 5：不同群使用不同 Agent

创建多个 SimpleChatAgent：

```python
"chat_group_1": {
    "enabled": True,
    "name": "群1助手",
    "class": "SimpleChatAgent",
    "config": {
        "monitored_groups": [1075786046],
        "system_prompt": "你是群1的助手..."
    }
},
"chat_group_2": {
    "enabled": True,
    "name": "群2助手",
    "class": "SimpleChatAgent",
    "config": {
        "monitored_groups": [123456789],
        "system_prompt": "你是群2的助手..."
    }
}
```

---

## 7. 配置验证

### 启动时检查

Bot 启动时会自动验证配置：

- ✅ 检查必需的环境变量
- ✅ 检查 NapCat 连接
- ✅ 检查 Agent 配置完整性
- ✅ 检查群号格式

### 常见配置错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `请在 .env 文件中配置 API Key` | 缺少环境变量 | 创建 `.env` 文件并填写 |
| `无法连接到 NapCat` | NapCat 未运行或配置错误 | 检查 NapCat 和端口配置 |
| `未知的 Agent 类型` | Agent 类名错误 | 检查 `class` 字段 |
| `Agent 不响应` | 群号或触发条件配置错误 | 检查 `monitored_groups` 和 `trigger_mode` |

---

## 8. 高级配置

### 8.1 自定义模型参数

```python
"config": {
    "model": "deepseek-v3",
    "temperature": 0.7,  # 需要修改 Agent 代码支持
    "max_tokens": 500
}
```

### 8.2 多语言支持

```python
"system_prompt": """You are a friendly assistant.
Respond in English."""
```

### 8.3 自定义日志级别

在 `utils/logger.py` 中修改：

```python
setup_logger(level="DEBUG")  # INFO, DEBUG, WARNING, ERROR
```

---

## 9. 配置最佳实践

### ✅ 推荐做法

1. **分离敏感信息**：API Key 放在 `.env` 中
2. **使用描述性名称**：Agent 名称清晰易懂
3. **合理设置触发模式**：根据群活跃度选择
4. **定期备份配置**：保存配置文件副本
5. **测试后再启用**：新 Agent 先在测试群测试

### ❌ 避免做法

1. **不要硬编码 API Key**：使用环境变量
2. **不要监听所有群**：只监听需要的群
3. **不要使用过于宽泛的关键词**：避免误触发
4. **不要忽略日志**：定期检查日志文件
5. **不要在生产环境直接修改**：先在测试环境验证

---

## 10. 配置更新

### 热更新（不支持）

当前版本不支持热更新配置，修改配置后需要重启 Bot：

```bash
# 停止 Bot (Ctrl+C)
# 修改配置文件
# 重新启动
python main.py
```

### 未来计划

- [ ] 支持配置热重载
- [ ] Web 配置界面
- [ ] 配置验证工具
- [ ] 配置导入/导出

---

## 11. 故障排查

### 配置相关问题

1. **Bot 不响应**
   - 检查 `monitored_groups` 是否包含目标群
   - 检查 `enabled` 是否为 `True`
   - 检查触发条件（@ 或关键词）

2. **Agent 未加载**
   - 查看启动日志中的 Agent 加载信息
   - 检查 `class` 字段是否正确
   - 检查是否有语法错误

3. **私聊消息发送失败**
   - 确认 `target_user` 配置正确
   - 检查 Bot 是否与目标用户是好友
   - 查看 NapCat 日志

---

## 12. 参考资料

- [README.md](README.md) - 项目说明
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [NapCat 文档](https://napneko.github.io/)
- [OneBot 11 协议](https://github.com/botuniverse/onebot-11)

---

**最后更新**：2026-02-04
