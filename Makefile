# 快速打包与部署
# 用法: make build | push | deploy | save | load
#
# 本地打包:   make build          # 或 make save 得到 qqbot.tar.gz
# 服务器部署: 见下方「服务器首次部署」说明

IMAGE := qqbot:latest
# 使用镜像仓库时设置，例如: export REGISTRY=registry.cn-hangzhou.aliyuncs.com/你的命名空间
REGISTRY ?=

.PHONY: build push deploy save load

# 本地构建镜像
build:
	docker compose build

# 构建并推送到镜像仓库（需先设置 REGISTRY 并 docker login）
push: build
	@if [ -z "$(REGISTRY)" ]; then echo "请设置 REGISTRY，如: make push REGISTRY=registry.cn-hangzhou.aliyuncs.com/ns"; exit 1; fi
	docker tag $(IMAGE) $(REGISTRY)/qqbot:latest
	docker push $(REGISTRY)/qqbot:latest

# 服务器上一键拉取并启动（不构建）
deploy:
	docker compose pull
	docker compose up -d

# 导出镜像为 tar.gz（无需仓库，用 scp 拷到服务器）
save: build
	docker save $(IMAGE) | gzip -c > qqbot.tar.gz
	@echo "已生成 qqbot.tar.gz，拷到服务器后执行: make load"

# 在服务器上从 tar.gz 加载镜像（与 save 配套）
load:
	@if [ ! -f qqbot.tar.gz ]; then echo "当前目录需要存在 qqbot.tar.gz"; exit 1; fi
	gunzip -c qqbot.tar.gz | docker load
	@echo "加载完成，执行 docker compose up -d 启动"

# 打印服务器部署步骤
server-help:
	@echo "--- 服务器首次部署 ---"
	@echo "方式一（tar 包，无需镜像仓库）："
	@echo "  1. 本机: make save"
	@echo "  2. 把 qqbot.tar.gz、docker-compose.yml、.env、config.yaml、bot.py、workflows/ 等拷到服务器"
	@echo "  3. 服务器: make load && docker compose up -d"
	@echo ""
	@echo "方式二（镜像仓库）："
	@echo "  1. 本机: make push REGISTRY=你的仓库地址"
	@echo "  2. 服务器放好 docker-compose.yml、.env、config.yaml 等，设 IMAGE_NAME=仓库/qqbot:latest"
	@echo "  3. 服务器: make deploy"
