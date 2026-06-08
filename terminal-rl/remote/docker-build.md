# 基于根目录 Dockerfile 构建镜像

Dockerfile 路径：`/Users/ruonan/Desktop/rollout-survey/OpenClaw-RL/Dockerfile`

镜像内容：Ubuntu 24.04 + Docker 29.1.3 + Compose 2.40.3 + uv + 已 checkout 到 `fix-terminal-rl` 分支的 OpenClaw-RL 仓库，启动时自动拉起 dockerd（DinD）。

## 构建

```bash
cd /Users/ruonan/Desktop/rollout-survey/OpenClaw-RL

# 基础构建
docker build -t openclaw-rl:dind .

# 推荐：启用 BuildKit + 走宿主网络（国内拉 apt/docker repo 稳一些）
DOCKER_BUILDKIT=1 docker build --network=host -t openclaw-rl:dind .
```

可选参数：

```bash
# 覆盖默认版本
docker build \
  --build-arg DOCKER_VERSION=29.1.3 \
  --build-arg COMPOSE_VERSION=2.40.3 \
  -t openclaw-rl:dind .

# 多架构（如需 arm64）
docker buildx build --platform linux/amd64,linux/arm64 -t openclaw-rl:dind --load .
```

## 运行（DinD 必须 `--privileged`）

参考根目录 `launch_docker.sh`：

```bash
docker run -d --name openclaw-rl \
  --privileged \
  --network host \
  --ipc host \
  --shm-size 64g \
  -v /data1:/data1 \
  -v /data0/ruonan/openclaw-dind-data:/var/lib/docker \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
  openclaw-rl:dind \
  tail -f /dev/null

docker exec -it openclaw-rl bash
```

## 进入容器后验证

```bash
docker version           # 应显示 29.1.3
docker compose version   # 应显示 2.40.3
docker info | grep -A6 "Registry Mirrors"
uv --version
ls /OpenClaw-RL && (cd /OpenClaw-RL && git branch --show-current)  # fix-terminal-rl
```

## 常见问题

- **构建时报找不到 `5:29.1.3-1~ubuntu.24.04~noble`**：去 `https://download.docker.com/linux/ubuntu/dists/noble/pool/stable/$(arch)/` 查看实际可用版本号，把 `--build-arg DOCKER_VERSION=` 改成存在的版本，或在 Dockerfile 里去掉版本锁。
- **`dockerd` 起不来**：先确认 run 时加了 `--privileged` 和 cgroup mount；查看容器内 `/tmp/dockerd.log`。
- **拉 docker.com GPG 慢/失败**：构建机器需能访问 `download.docker.com`，必要时加 `--build-arg HTTPS_PROXY=...` 并在 Dockerfile 中 `ENV HTTPS_PROXY` 透传。
