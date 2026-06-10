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
    && PKG_VER="5:${DOCKER_VERSION}-1~ubuntu.24.04~noble" \
    && CLI_VER="1:${DOCKER_VERSION}-1~ubuntu.24.04~noble" \
    && COMPOSE_PKG="${COMPOSE_VERSION}-1~ubuntu.24.04~noble" \
    && apt-get install -y --no-install-recommends \
        docker-ce=${PKG_VER} \
        docker-ce-cli=${CLI_VER} \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin=${COMPOSE_PKG} \
    && rm -rf /var/lib/apt/lists/*

# ---- docker daemon config -----------------------------------------------------
RUN mkdir -p /etc/docker
COPY daemon.json /etc/docker/daemon.json

# ---- uv -----------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# ---- OpenClaw-RL repo @ fix-terminal-rl --------------------------------------
RUN git clone --branch fix-terminal-rl --single-branch \
        https://github.com/rnwang04/OpenClaw-RL.git /OpenClaw-RL

# ---- entrypoint: start dockerd then drop into shell --------------------------
COPY dind-entrypoint.sh /usr/local/bin/dind-entrypoint.sh
RUN chmod +x /usr/local/bin/dind-entrypoint.sh

WORKDIR /OpenClaw-RL
ENTRYPOINT ["/usr/local/bin/dind-entrypoint.sh"]
CMD ["bash"]
