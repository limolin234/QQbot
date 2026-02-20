# ==========================================
# 第一阶段：构建环境 (必须起名为 builder)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装构建依赖（如果某些 Python 包需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖到指定目录
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 拷贝源代码
COPY . .

# ==========================================
# 第二阶段：运行环境 (Distroless 极致安全镜像)
# ==========================================
FROM gcr.io/distroless/python3-debian12:latest

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    PYTHONPATH=/usr/local/lib/python3.11/site-packages

WORKDIR /app

# 1. 拷贝库文件
COPY --from=builder /install /usr/local

# 2. 【核心修复】拷贝代码并更改所有者
# Distroless 默认 nonroot 用户 ID 是 65532
# 必须在 COPY 时通过 --chown 授权，否则 nonroot 用户无法在 /app 下创建 ./logs
COPY --from=builder --chown=65532:65532 /app /app

# 3. 切换到安全用户
USER 65532

# 4. 启动程序
ENTRYPOINT ["python3", "main.py"]
