# ==========================================
# 第一阶段：构建者 (Builder)
# ==========================================
FROM python:3.11-slim AS builder

# 设置构建环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 安装构建依赖（如果你的 requirements 包含需要编译的库，如 ujson, cryptography 等）
# Distroless 不含编译器，所以必须在这里完成所有安装
COPY requirements.txt .

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# 拷贝源代码
COPY . .

# ==========================================
# 第二阶段：运行环境 (Runtime - Distroless)
# ==========================================
# 使用 Google 提供的 Python 3 Distroless 镜像 (基于 Debian 12)
FROM gcr.io/distroless/python3-debian12:latest

# 维护元数据
LABEL maintainer="YourName" \
      description="Agent Service with Distroless Security"

# 设置运行时环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    # 关键：将第一阶段安装的库路径加入 Python 搜索路径
    PYTHONPATH=/usr/local/lib/python3.11/site-packages

WORKDIR /app

# 1. 从构建阶段拷贝安装好的第三方库
COPY --from=builder /install /usr/local

# 2. 从构建阶段拷贝时区数据 (Distroless 默认不含时区文件)
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# 3. 拷贝应用程序代码
COPY --from=builder /app /app

# 4. 安全约束：Distroless 默认以 non-root 用户运行（UID 65532）
# 如果你的程序需要写日志到特定目录，请确保该目录权限正确
USER 65532

# 5. 启动程序
# 注意：Distroless 没有 shell，必须使用 JSON 数组格式 (Exec Form)
ENTRYPOINT ["python3", "main.py"]
