# 配置说明文档

本文档详细说明了 QQ Bot 的所有可配置参数。

## 📁 配置文件位置

### 1. `.env` - 环境变量配置
存放敏感信息（API Key 等）

### 2. `config/bot_config.py` - Bot 行为配置
存放 Bot 的行为参数、模型配置、群号等

### 3. `config/napcat_config.py` - NapCat 连接配置
存放 NapCat WebSocket 连接地址

---

## 🔧 详细配置说明

### 一、环境变量配置 (`.env`)

```env
# LLM API 配置
YUNWU_API_KEY = sk-xxx...
API_BASE_URL = https://yunwu.ai/v1
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `YUNWU_API_KEY` | LLM API 密钥 | `sk-TplxUzAgXpma6pUb...` |
| `API_BASE_URL` | LLM API 基础地址 | `https://yunwu.ai/v1` |

**注意**：
- 不需要配置 NapCat 相关的 Token
- 支持任何 OpenAI Compatible API

---

### 二、Bot 行为配置 (`config/bot_config.py`)

#### 2.1 基础配置

```python
# 监听的群号列表
MONITORED_GROUPS = [1075786046]

# Bot QQ 号（可选，如果不设置会自动获取）
BOT_QQ = None
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `MONITORED_GROUPS` | `list[int]` | 要监听的 QQ 群号列表 | `[1075786046]` |
| `BOT_QQ` | `int` or `None` | Bot 的 QQ 号（可选） | `None`（自动获取） |

**如何修改**：
```python
# 监听多个群
MONITORED_GROUPS = [1075786046, 123456789, 987654321]

# 手动指定 Bot QQ 号（通常不需要）
BOT_QQ = 3014249817
```

---

#### 2.2 Agent 配置

```python
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
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `MODEL_ID` | `str` | LLM 模型 ID | `"deepseek-v3"` |
| `AGENT_SYSTEM_PROMPT` | `str` | Agent 的系统提示词 | 见上方 |

**如何修改**：

**修改模型**：
```python
# 使用 GPT-4
MODEL_ID = "gpt-4"

# 使用 DeepSeek V3
MODEL_ID = "deepseek-v3"

# 使用其他模型
MODEL_ID = "your-model-name"
```

**修改系统提示词**：
```python
# 专业助手
AGENT_SYSTEM_PROMPT = """你是一个专业的技术助手，擅长编程和技术问题。
回答要准确、详细，提供代码示例。"""

# 幽默助手
AGENT_SYSTEM_PROMPT = """你是一个幽默风趣的群聊助手。
回答要轻松有趣，适当使用表情和网络用语。"""

# 特定领域助手
AGENT_SYSTEM_PROMPT = """你是一个 Python 编程助手。
专注于回答 Python 相关问题，提供代码示例和最佳实践。"""
```

---

#### 2.3 唤醒配置

```python
# 唤醒关键词（正则表达式）
WAKE_WORDS = [
    r"小助手",
    r"bot",
    r"机器人",
    r"助手"
]
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `WAKE_WORDS` | `list[str]` | 唤醒关键词列表（支持正则） | 见上方 |

**如何修改**：
```python
# 添加更多唤醒词
WAKE_WORDS = [
    r"小助手",
    r"bot",
    r"机器人",
    r"助手",
    r"AI",
    r"智能助手",
    r"帮我"
]

# 使用正则表达式（更灵活）
WAKE_WORDS = [
    r"小助手",
    r"bot|机器人|助手",  # 匹配任意一个
    r"帮我.*",  # 匹配"帮我"开头的任何内容
]
```

**注意**：
- 支持 Python 正则表达式语法
- 除了关键词，@ bot 也会触发响应

---

#### 2.4 安全配置

```python
# 最大消息长度（字符）
MAX_MESSAGE_LENGTH = 500

# 每分钟最大消息数
MAX_MESSAGES_PER_MINUTE = 10

# 违规次数阈值（超过此次数将暂停服务）
VIOLATION_THRESHOLD = 3
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `MAX_MESSAGE_LENGTH` | `int` | 单条消息最大字符数 | `500` |
| `MAX_MESSAGES_PER_MINUTE` | `int` | 每分钟最多发送消息数 | `10` |
| `VIOLATION_THRESHOLD` | `int` | 违规次数阈值 | `3` |

**如何修改**：
```python
# 允许更长的消息
MAX_MESSAGE_LENGTH = 1000

# 提高发送频率（谨慎调整）
MAX_MESSAGES_PER_MINUTE = 20

# 更严格的违规处理
VIOLATION_THRESHOLD = 2
```

**安全机制说明**：
1. 消息长度超过 `MAX_MESSAGE_LENGTH` 时不会发送
2. 每分钟发送超过 `MAX_MESSAGES_PER_MINUTE` 条时不会发送
3. 累计违规达到 `VIOLATION_THRESHOLD` 次后，该群服务将被暂停

---

### 三、NapCat 连接配置 (`config/napcat_config.py`)

```python
# WebSocket 服务端地址（正向 WS）
NAPCAT_WS_URL = "ws://127.0.0.1:3001"

# HTTP API 地址（可选）
NAPCAT_HTTP_URL = "http://127.0.0.1:3000"
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `NAPCAT_WS_URL` | `str` | WebSocket 连接地址 | `"ws://127.0.0.1:3001"` |
| `NAPCAT_HTTP_URL` | `str` | HTTP API 地址 | `"http://127.0.0.1:3000"` |

**如何修改**：
```python
# 修改端口（如果在 NapCat WebUI 中修改了端口）
NAPCAT_WS_URL = "ws://127.0.0.1:8080"

# 远程连接（不推荐，有安全风险）
NAPCAT_WS_URL = "ws://192.168.1.100:3001"
```

---

## 🎯 常见配置场景

### 场景 1：监听多个群

```python
# config/bot_config.py
MONITORED_GROUPS = [
    1075786046,  # 测试群
    123456789,   # 工作群
    987654321    # 学习群
]
```

### 场景 2：使用不同的 LLM

```python
# config/bot_config.py
MODEL_ID = "gpt-4o-mini"  # 使用 GPT-4o-mini

# .env
API_BASE_URL = https://api.openai.com/v1
YUNWU_API_KEY = sk-xxx...  # OpenAI API Key
```

### 场景 3：自定义 Bot 人格

```python
# config/bot_config.py
AGENT_SYSTEM_PROMPT = """你是一个二次元风格的可爱助手，名叫小樱。

你的特点：
- 说话可爱，喜欢用"呢"、"哦"等语气词
- 偶尔使用颜文字 (๑•̀ㅂ•́)و✧
- 对技术问题很专业，但表达方式轻松活泼
- 记得用户的名字，像朋友一样交流

请用中文回复，保持可爱和专业的平衡～"""
```

### 场景 4：严格的安全限制

```python
# config/bot_config.py
MAX_MESSAGE_LENGTH = 300  # 更短的消息
MAX_MESSAGES_PER_MINUTE = 5  # 更低的频率
VIOLATION_THRESHOLD = 1  # 一次违规就暂停
```

---

## 📝 配置修改后的操作

修改配置后需要：

1. **重启 Bot**：
   ```bash
   # 停止 Bot (Ctrl+C)
   # 重新启动
   python main.py
   ```

2. **检查日志**：
   ```bash
   tail -f logs/bot.log
   ```

3. **测试功能**：
   - 在群中发送测试消息
   - 检查 Bot 是否正常响应

---

## ⚠️ 注意事项

1. **不要泄露 API Key**：`.env` 文件已在 `.gitignore` 中，不会被提交到 Git
2. **谨慎调整安全参数**：过高的频率可能导致账号风险
3. **系统提示词很重要**：它决定了 Bot 的行为和风格
4. **正则表达式要测试**：错误的正则可能导致无法唤醒或误触发

---

## 🔍 配置验证

启动 Bot 时会显示当前配置：

```
2026-02-04 15:57:17 | INFO | SimpleChatAgent 初始化完成，模型: deepseek-v3
2026-02-04 15:57:17 | DEBUG | 系统提示词: 你是一个友好、乐于助人的 QQ 群助手...
```

检查这些信息确认配置是否正确。
