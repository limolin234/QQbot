# Config Studio 运行说明

## 1. 后端启动
在仓库根目录执行：

```bash
/home/yunzechen/Code/QQbot/.venv/bin/python -m tools.config_studio.server
```

默认监听：127.0.0.1:8787

关键接口：
- GET /api/health
- GET /api/config/all
- GET /api/config/agent/raw
- PUT /api/config/env
- PUT /api/config/agent
- PUT /api/config/agent/raw
- POST /api/scheduler/compile
- POST /api/scheduler/timeline
- POST /api/deploy/test-connection
- POST /api/deploy/push
- GET /api/history
- POST /api/history/restore

## 2. 前端启动
```bash
cd tools/config-studio-web
npm install
npm run dev
```

默认访问：http://127.0.0.1:4173

## 3. 已实现能力
- .env 可视化编辑并自动保存（LLM_API_KEY, LLM_API_BASE_URL）
- agent_config.yaml section 编辑
- 原始 YAML 高级模式
- Scheduler 可视化树编辑（action/group/if）
- Scheduler 编译与时间线预览
- 推送与自动重启（SFTP + SSH）
- 快照查看与恢复

## 4. 自测命令
后端自测：

```bash
/home/yunzechen/Code/QQbot/.venv/bin/python -m tools.config_studio.server.self_test
```

前端构建：

```bash
cd tools/config-studio-web
npm run build
```
