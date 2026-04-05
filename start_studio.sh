#!/bin/bash

# 发生错误时退出
set -e

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "========================================="
echo "启动 Config Studio (前端 + 后端)"
echo "========================================="

# 内部清理函数，用于在关闭时清理后台进程
cleanup() {
    echo ""
    echo "🔴 正在停止 Config Studio..."
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "✅ 已关闭后端服务 (PID: $BACKEND_PID)"
    fi
    exit 0
}

# 捕获 Ctrl+C (SIGINT) 和终止信号 (SIGTERM)
trap cleanup SIGINT SIGTERM

# ================= 1. 启动后端 =================
echo "🚀 1. 正在启动后台 API 服务..."
PYTHON_EXEC="$DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "❌ 错误: 未检测到虚拟环境中的 Python: $PYTHON_EXEC"
    echo "请确认是否已创建虚拟环境并安装依赖。"
    exit 1
fi

cd "$DIR"
# 后台运行后端服务
$PYTHON_EXEC -m tools.config_studio.server &
BACKEND_PID=$!
echo "✅ 后端已启动 (PID: $BACKEND_PID, 默认监听 127.0.0.1:8787)"

# 稍微等待一秒，确保后端报错能及时输出
sleep 1 

# ================= 2. 启动前端 =================
echo "🚀 2. 正在启动前端 Web 服务..."
WEB_DIR="$DIR/tools/config-studio-web"

if [ ! -d "$WEB_DIR" ]; then
    echo "❌ 错误: 未找到前端目录 $WEB_DIR"
    cleanup
fi

# 检查 Node 和 npm
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "❌ 错误: 未安装 Node.js 或 npm，请先安装 Node.js"
    cleanup
fi

cd "$WEB_DIR"

if [ ! -d "node_modules" ]; then
    echo "📦 未检测到 node_modules，正在运行 npm install 安装依赖..."
    npm install
fi

echo "========================================="
echo "🎉 启动完成！"
echo "前端页面请访问: http://127.0.0.1:4173"
echo "注: 按下 Ctrl+C 会同时安全停止前后端服务"
echo "========================================="

# 前台运行 npm
npm run dev

# 如果 npm 因任何原因退出（例如发生错误），执行清理操作
cleanup
