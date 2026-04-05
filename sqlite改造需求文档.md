# SQLite 改造需求文档（架构师版）

## 目标与结论

- 目标：将消息上下文存储从文件日志机制升级为 SQLite，解决高并发下文件锁冲突与截断复杂度问题。
- 结论：是，运行时主存储将由 SQLite 替代 message.jsonl。
- 兼容策略：message.jsonl 仅保留为可选迁移输入或导出产物，不再作为运行时读取依赖。

## 背景问题

- 当前消息链路以 message.jsonl 为核心存储，写入后按行数阈值截断。
- 在容器挂载与并发读写下，文件 I/O（尤其截断、锁、flush/fsync）复杂且脆弱。
- summary 流程仍直接读取 LOG_FILE_PATH，存在与新存储方向不一致的问题。

## 设计原则（Minimal Invasive）

- 保持对上层业务调用接口稳定：尽量不改动调用方业务语义。
- 先引入 SQLite 适配层，再逐步替换直接文件读取代码。
- 不做无关重构，不改变 auto_reply、dida、summary 的核心策略逻辑。

## 上下文剪裁（Context Pruning）

Coder 只允许优先阅读并修改下列文件：

1. workflows/message_observe.py
2. workflows/summary.py
3. workflows/auto_reply.py
4. workflows/dida/dida_agent.py
5. main.py（仅在初始化调用需要时）

Coder 明确禁止改动：

1. workflows/agent_observe.py（除非要抽取共用 DB 工具，默认不动）
2. workflows/scheduler/**
3. bot.py
4. agent_pool.py
5. deploy/**

## 依赖分析（Dependency Analysis）

- Python 内置 sqlite3 可满足当前需求，无需新增第三方依赖。
- 可选优化库：aiosqlite（本期不引入，避免侵入和行为变化）。
- 与现有依赖兼容性：无冲突风险。

## 数据模型设计（Data Flow + Schema）

### 新数据库文件

- 路径建议：data/message_store.db
- 建议开启 WAL：提高并发读写能力。

### 表结构（伪 SQL）

```sql
CREATE TABLE IF NOT EXISTS message_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  chat_type TEXT NOT NULL,
  group_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  user_name TEXT NOT NULL,
  cleaned_message TEXT NOT NULL,
  raw_message TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_msg_chat_group_ts
  ON message_logs(chat_type, group_id, ts);

CREATE INDEX IF NOT EXISTS idx_msg_chat_user_ts
  ON message_logs(chat_type, user_id, ts);

CREATE INDEX IF NOT EXISTS idx_msg_ts
  ON message_logs(ts);
```

### 裁剪策略

- 从“按文件行数截断”变更为“按数据库保留条数清理”。
- 建议阈值：保留最近 N 条（例如 50000），超过后按 id/ts 删除旧数据。

## 接口定义

### 新增存储接口（建议放在 workflows/message_observe.py 内，后续可抽文件）

```python
def db_start_up() -> None:
    """初始化数据库、建表、建索引、PRAGMA。"""

def insert_message(
    *,
    ts: str,
    chat_type: str,
    group_id: str,
    user_id: str,
    user_name: str,
    cleaned_message: str,
    raw_message: str = "",
) -> None:
    """写入一条消息记录。"""

def cleanup_old_messages(*, max_rows: int) -> int:
    """删除超出保留上限的旧记录，返回删除行数。"""

def query_recent_context_messages(
    *,
    chat_type: str,
    group_id: str,
    user_id: str,
    current_ts: str,
    current_cleaned_message: str,
    limit: int,
    max_chars: int,
    window_seconds: int,
) -> list[str]:
    """按会话范围查询上下文，输出与现有 load_recent_context_messages 一致。"""

def query_lines_for_summary(*, since_ts: str | None = None) -> list[str]:
    """供 summary 使用，输出兼容旧逻辑的行文本或结构对象。"""
```

### 现有接口兼容要求

- 保留 `group_entrance` / `private_entrance` 签名不变。
- 保留 `load_recent_context_messages` 对外函数名不变：内部改为查询 SQLite。
- summary 不再直接 `open(LOG_FILE_PATH)`，改为调用统一查询函数。

## 影响范围分析

- 主要修改：
  - workflows/message_observe.py
  - workflows/summary.py
- 低风险适配：
  - workflows/auto_reply.py（若仅依赖函数签名则可不改）
  - workflows/dida/dida_agent.py（同上）
- 初始化链路：
  - main.py（如需在 startup 中显式初始化 DB）
- 文档：
  - README.md（新增数据存储说明）

## 实施步骤（Task List）

1. [Step 1] 在 message_observe 中新增 SQLite 初始化函数（建库、建表、索引、WAL）。
2. [Step 2] 将 `_write_log` 的文件写入替换为 `insert_message`。
3. [Step 3] 将 `_truncate_log` 替换为 `cleanup_old_messages`，删除文件截断逻辑。
4. [Step 4] 将 `load_recent_context_messages` 改为 SQL 查询实现，保持返回格式不变。
5. [Step 5] 修改 summary 的日志读取入口：由文件读取改为 DB 查询接口。
6. [Step 6] 保留 message.jsonl 迁移脚本（一次性导入旧数据，可选）。
7. [Step 7] 补充最小测试：写入、上下文查询、清理策略、summary 数据源兼容。
8. [Step 8] 更新 README 与部署说明（DB 文件路径、备份策略）。

## 数据流说明

1. 收到消息 -> `group_entrance/private_entrance`。
2. 清洗消息 -> `insert_message` 写入 SQLite。
3. auto_reply / dida -> 调用 `load_recent_context_messages`（内部 SQL 查询）。
4. summary 定时任务 -> 从 SQLite 拉取窗口数据并聚合。
5. 后台定期执行 `cleanup_old_messages`。

## 非目标（本期不做）

- 不改动 AutoReply / Dida 的判定模型与提示词逻辑。
- 不引入 ORM（例如 SQLAlchemy）。
- 不新增网络服务或外部数据库。

## 验收标准（Definition of Done）

- 运行时不再依赖 message.jsonl 作为上下文主数据源。
- message_observe 中不再有 open(message.jsonl) 读写主路径。
- summary 不再直接读取 LOG_FILE_PATH 文件。
- 压测下无 `Device or resource busy` 文件替换错误。
- auto_reply / dida 的上下文行为与迁移前一致（内容等价或更稳定）。

## 风险与回滚

- 风险：历史 message.jsonl 未迁移导致短期上下文减少。
- 规避：提供一次性导入脚本（jsonl -> sqlite）。
- 回滚：保留 feature flag（如 `MESSAGE_STORE_BACKEND=jsonl|sqlite`），紧急时切回旧路径。

## 架构决策（最终回答）

- 是的，本方案定义为“SQLite 替代 message.jsonl 作为运行时主存储”。
- message.jsonl 在改造后只作为迁移输入或调试导出，不再是业务关键路径。
