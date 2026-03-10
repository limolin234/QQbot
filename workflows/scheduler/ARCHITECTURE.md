# Unified Scheduler Architecture

## 1. Overview
The Unified Scheduler aims to centralize all timed tasks (cron and interval-based) into a single module driven by a YAML configuration. This replaces scattered hardcoded loops and `aiocron` calls, improving maintainability and observability.

## 2. Directory Structure
New directory: `workflows/scheduler/`

```text
workflows/scheduler/
├── __init__.py           # Exposes SchedulerManager
├── manager.py            # SchedulerManager implementation
├── registry.py           # ActionRegistry implementation
├── actions/              # Built-in actions
│   ├── __init__.py
│   ├── core.py           # Basic actions (send_msg, etc.)
│   ├── dida.py           # Dida wrappers
│   └── summary.py        # Summary wrappers
└── config_models.py      # Pydantic models for validation
```

## 3. Core Components

### 3.1 Configuration (YAML)
We will extend `workflows/agent_config.yaml` with a `scheduler_config` section.

```yaml
scheduler_config:
  file_name: scheduler_manager.py  # Virtual filename for loader compatibility
  config:
    timezone: "Asia/Shanghai"
    schedules:
      - name: "Morning Greeting"
        type: "cron"
        expression: "0 8 * * *"
        steps:
          - action: "core.send_group_msg"
            params:
              group_id: "123456"
              message: "Good morning!"
      
      - name: "Task Polling"
        type: "interval"
        seconds: 60
        steps:
          - action: "dida.poll"
```

### 3.2 Action Registry (`registry.py`)
A central registry mapping string IDs to Python callables.

**Interface:**
```python
# Type definition
ActionFunc = Callable[..., Awaitable[Any]]

class ActionRegistry:
    def register(self, action_id: str, func: ActionFunc) -> None:
        """Register a new action."""
        
    def get(self, action_id: str) -> ActionFunc:
        """Retrieve an action by ID. Raises ValueError if not found."""
        
    def list_actions(self) -> List[str]:
        """List all registered action IDs."""
```

### 3.3 Scheduler Manager (`manager.py`)
The main engine responsible for:
1.  Loading configuration.
2.  Initializing `aiocron` for Cron jobs.
3.  Managing `asyncio` loops for Interval jobs.
4.  Executing `steps` sequentially.

**Key Logic:**
-   **Step Execution**: Iterate through `steps`. If a step fails, log error. Support `fail_fast` (optional future feature).
-   **Context**: Pass a context object to actions if needed (e.g., shared state between steps), though initially actions will be stateless.

## 4. Interfaces & Data Flow

### 4.1 Action Signature
All actions must be async functions accepting keyword arguments.

```python
async def send_group_msg(group_id: str, message: str) -> None:
    ...

async def dida_poll(project_ids: List[str] = None) -> None:
    ...
```

### 4.2 Integration with Existing Modules
-   **Summary**: Refactor `workflows/summary.py` to expose a `run_summary(mode: str)` async function.
-   **Dida**: Refactor `workflows/dida/dida_scheduler.py` logic into `workflows/scheduler/actions/dida.py` or expose a single poll function in `dida_service.py`.

## 5. Implementation Plan

### Phase 1: Infrastructure (Core)
1.  Create `workflows/scheduler/` directory.
2.  Implement `ActionRegistry` and `SchedulerManager`.
3.  Define Pydantic models in `config_models.py`.
4.  Implement `core` actions (`send_group_msg`, `send_private_msg`).

### Phase 2: Migration (Feature Parity)
1.  **Migrate Summary**:
    -   Register `summary.daily_report` action.
    -   Update `agent_config.yaml` to include the summary schedule.
    -   Remove `aiocron` code from `workflows/summary.py`.
2.  **Migrate Dida**:
    -   Register `dida.poll` and `dida.push_task_list`.
    -   Update `agent_config.yaml`.
    -   Deprecate `workflows/dida/dida_scheduler.py` loop.

### Phase 3: Integration
1.  Modify `bot.py` to initialize `SchedulerManager` on startup.
2.  Verify all tasks run as expected.

## 6. Migration & Compatibility Strategy (Zero Downtime)

To ensure existing users' setups do not break, we will adopt a **"Dual-Mode"** strategy.

### 6.1 Logic
The system will detect whether `scheduler_config` exists in `agent_config.yaml`.

-   **Legacy Mode (Default)**:
    -   If `scheduler_config` is **missing**:
        -   `summary.py` continues to use its internal `aiocron` logic.
        -   `dida_scheduler.py` continues to use its internal `while True` loop.
        -   The `SchedulerManager` initializes but stays idle or logs a warning.
-   **Modern Mode**:
    -   If `scheduler_config` is **present**:
        -   `SchedulerManager` takes over all scheduling.
        -   `summary.py` detects this and **skips** initializing its internal cron.
        -   `dida_scheduler.py` detects this and **skips** starting its internal loop.

### 6.2 Implementation Details
1.  **Shared Flag**: Create a helper `is_scheduler_enabled()` in `agent_config_loader.py`.
2.  **Conditional Startup**:
    -   In `summary.py`:
        ```python
        if not is_scheduler_enabled():
            # Start legacy aiocron
            ...
        ```
    -   In `dida_scheduler.py`:
        ```python
        if not is_scheduler_enabled():
            # Start legacy loop
            ...
        ```

### 6.3 User Update Path
1.  Users pull the latest code.
2.  By default, nothing changes (Legacy Mode).
3.  To upgrade, users copy `scheduler_config` from `.example` to their `agent_config.yaml`.
4.  Upon restart, the system automatically switches to Modern Mode.

## 7. Impact Analysis
-   **Files to Modify**:
    -   `workflows/agent_config.yaml` (Add config)
    -   `bot.py` (Add startup logic)
    -   `workflows/summary.py` (Add conditional check)
    -   `workflows/dida/dida_scheduler.py` (Add conditional check)
-   **New Files**:
    -   `workflows/scheduler/*`

## 8. Dependencies
-   `aiocron`: Already in project.
-   `pydantic`: Already in project.
-   `asyncio`: Standard lib.