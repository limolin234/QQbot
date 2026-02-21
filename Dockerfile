# ==========================================
# 第一阶段：构建阶段 (Builder)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# 1. 升级基础工具，确保安装过程安全
RUN pip install --no-cache-dir --upgrade pip setuptools wheel jaraco.context

# 2. 安装依赖到指定目录 (--user 会安装到 /root/.local)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ==========================================
# 第二阶段：运行阶段 (Final)
# ==========================================
FROM python:3.11-slim

# 1. 设置环境变量
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 将 Python 寻找包的路径指向从 builder 拷贝过来的目录
    PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/root/.local/lib/python3.11/site-packages

WORKDIR /app

# 2. 从构建阶段只拷贝安装好的 Python 包
# 这一步非常关键：它避开了 builder 镜像中可能存在的旧版 wheel 元数据
COPY --from=builder /root/.local /root/.local

# 3. 拷贝项目文件
COPY . .

# 4. 权限与目录处理
# 显式创建需要的文件夹并设置权限
RUN mkdir -p /app/data /app/logs && \
    chmod -R 755 /app

# 5. 启动命令
CMD ["python", "main.py"]
