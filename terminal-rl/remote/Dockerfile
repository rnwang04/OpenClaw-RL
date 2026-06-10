# Ubuntu 24.04.4 LTS (noble) with Docker-in-Docker + uv + OpenClaw-RL checkout
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ---- base packages ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        git \
        sudo \
        iproute2 \
        iptables \
        kmod \
        procps \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ---- docker 29.1.3 + compose 2.40.3 ------------------------------------------
ARG DOCKER_VERSION=29.1.3
ARG COMPOSE_VERSION=2.40.3
ARG DOCKER_PACKAGE_EPOCH=5
ARG DOCKER_APT_URL=https://download.docker.com/linux/ubuntu
ARG DOCKER_GPG_URL=
RUN install -m 0755 -d /etc/apt/keyrings \
    && DOCKER_GPG_URL="${DOCKER_GPG_URL:-${DOCKER_APT_URL}/gpg}" \
    && curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors \
        -o /tmp/docker.gpg.asc "${DOCKER_GPG_URL}" \
    && gpg --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker.gpg.asc \
    && rm -f /tmp/docker.gpg.asc \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        ${DOCKER_APT_URL} noble stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && PKG_VER="${DOCKER_PACKAGE_EPOCH}:${DOCKER_VERSION}-1~ubuntu.24.04~noble" \
    && CLI_VER="${DOCKER_PACKAGE_EPOCH}:${DOCKER_VERSION}-1~ubuntu.24.04~noble" \
    && COMPOSE_PKG="${COMPOSE_VERSION}-1~ubuntu.24.04~noble" \
    && apt-get install -y --no-install-recommends \
        docker-ce=${PKG_VER} \
        docker-ce-cli=${CLI_VER} \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin=${COMPOSE_PKG} \
    && rm -rf /var/lib/apt/lists/*

# ---- docker daemon config -----------------------------------------------------
RUN mkdir -p /etc/docker \
    && printf '%s\n' \
        '{' \
        '  "default-address-pools": [' \
        '    { "base": "10.200.0.0/12", "size": 24 },' \
        '    { "base": "172.28.0.0/14", "size": 24 }' \
        '  ],' \
        '  "registry-mirrors": [' \
        '    "https://docker.xuanyuan.dev",' \
        '    "https://docker.1ms.run",' \
        '    "https://hub.rat.dev",' \
        '    "https://docker.m.daocloud.io"' \
        '  ]' \
        '}' \
        > /etc/docker/daemon.json

# ---- uv -----------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# ---- OpenClaw-RL repo @ fix-terminal-rl --------------------------------------
RUN git clone --branch fix-terminal-rl --single-branch \
        https://github.com/rnwang04/OpenClaw-RL.git /OpenClaw-RL

# ---- entrypoint: start dockerd then drop into shell --------------------------
RUN printf '%s\n' \
        '#!/bin/sh' \
        'set -e' \
        '' \
        'rm -rf /var/run/docker /run/docker /var/run/containerd /run/containerd' \
        'mkdir -p /var/run /run' \
        '' \
        'nohup dockerd \' \
        '    --config-file=/etc/docker/daemon.json \' \
        '    --host=unix:///var/run/docker.sock \' \
        '    > /tmp/dockerd.log 2>&1 &' \
        '' \
        '# wait for socket' \
        'for i in $(seq 1 30); do' \
        '    [ -S /var/run/docker.sock ] && break' \
        '    sleep 1' \
        'done' \
        '' \
        'exec "$@"' \
        > /usr/local/bin/dind-entrypoint.sh \
    && chmod +x /usr/local/bin/dind-entrypoint.sh

WORKDIR /OpenClaw-RL
ENTRYPOINT ["/usr/local/bin/dind-entrypoint.sh"]
CMD ["bash"]
