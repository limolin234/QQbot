# ==========================================
# 第一阶段：构建
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 拷贝代码
COPY . .

# 【关键】在镜像内部预先创建好需要写入的文件和文件夹
RUN mkdir -p /app/logs && touch /app/message.jsonl

# ==========================================
# 第二阶段：运行 (Distroless)
# ==========================================
FROM gcr.io/distroless/python3-debian12:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    PYTHONPATH=/usr/local/lib/python3.11/site-packages

WORKDIR /app

# 拷贝库
COPY --from=builder /install /usr/local

# 【核心修复】使用 --chown 将整个 /app 目录授权给 nonroot 用户 (65532)
# 这样程序在运行时可以自由读写 /app/logs 和 /app/message.jsonl
COPY --from=builder --chown=65532:65532 /app /app

USER 65532

ENTRYPOINT ["python3", "main.py"]
