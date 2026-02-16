# Makefile 使用指南

项目根目录的 `Makefile` 用于 Docker 打包与服务器部署。

## 命令一览

| 命令 | 说明 |
|------|------|
| `make build` | 本地构建镜像（`docker compose build`） |
| `make save` | 构建并导出为 `qqbot.tar.gz`，拷到服务器用 |
| `make load` | 在服务器上从 `qqbot.tar.gz` 加载镜像（需先有该文件） |
| `make push` | 构建并推送到镜像仓库（需设 `REGISTRY=...` 且已 `docker login`） |
| `make deploy` | 服务器上一键拉取镜像并启动（需已配置 `IMAGE_NAME`） |
| `make server-help` | 打印服务器部署步骤说明 |

## 本地打包

```bash
make build    # 仅构建
make save     # 构建并生成 qqbot.tar.gz
```

## 服务器部署

### 方式一：tar 包（无需镜像仓库）

1. 本机：`make save`
2. 将 **qqbot.tar.gz** 以及 **docker-compose.yml、.env、config.yaml、bot.py、workflows/** 等拷到服务器同一目录
3. 服务器：`make load && docker compose up -d`

> 使用 `make load` 时保持 **qqbot.tar.gz 为单个压缩文件**，不要解压。

### 方式二：镜像仓库

1. 本机：`make push REGISTRY=你的仓库地址`（需先 `docker login`）
2. 服务器放好 docker-compose 与配置，设置 `IMAGE_NAME=仓库/qqbot:latest`
3. 服务器：`make deploy`

## 运行后常用命令

- 看日志：`docker compose logs -f qqbot`
- 重启：`docker compose restart`
- 停止：`docker compose down`
