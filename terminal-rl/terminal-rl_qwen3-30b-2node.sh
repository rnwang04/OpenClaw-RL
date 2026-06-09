#!/usr/bin/env bash
set -euo pipefail
set -x

# Two-node version of terminal-rl_qwen3-30b.sh
#
# Topology:
#   Node 0 (NODE_ROLE=head)   : actor  | 8 GPUs | TP=4 EP=4 DP=2
#   Node 1 (NODE_ROLE=worker) : sglang | 8 GPUs | 4 engines x TP=2
#
# Launch (see footer comments for full step-by-step):
#   On head   :  HEAD_IP=<head_ip> NODE_ROLE=head   bash terminal-rl/terminal-rl_qwen3-30b-2node.sh
#   On worker :  HEAD_IP=<head_ip> NODE_ROLE=worker bash terminal-rl/terminal-rl_qwen3-30b-2node.sh
#
# Required env (both nodes, must agree):
#   HEAD_IP                : reachable IP of the head node (used as ray --address and MASTER_ADDR)
#   CONDA_ENV_PATH         : conda env prefix containing python + ray + sglang + megatron
#   HF_CKPT / REF_LOAD     : model paths (same path on both nodes — NFS or local mirror)
#   SAVE_CKPT              : checkpoint save dir (head node)
#   ROLLOUT_PROMPT_DATA    : training jsonl path (head node)
#
# Optional env:
#   RAY_HEAD_PORT          : ray GCS port (default 6379)
#   RAY_DASHBOARD_PORT     : ray dashboard port (default 8265)
#   RAY_TMPDIR             : ray temp dir (default /tmp/ray)
#   WORKER_URLS            : comma-list of remote env worker URLs (router will dispatch)

log() { echo "[$(date +'%F %T')] $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing cmd: $1"; exit 1; }; }

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ─── Topology constants ────────────────────────────────────────────────────────
NODE_ROLE="${NODE_ROLE:?NODE_ROLE must be 'head' or 'worker'}"
HEAD_IP="${HEAD_IP:?HEAD_IP must be set to the head node reachable IP}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"

# Each node has 8 GPUs.
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# Actor goes on 1 node (8 GPUs), rollout goes on 1 node (8 GPUs).
ACTOR_GPUS_PER_NODE=8
ACTOR_NUM_NODES=1
ROLLOUT_NUM_GPUS=8
ROLLOUT_NUM_GPUS_PER_ENGINE=2   # 8 / 2 = 4 sglang engines, each TP=2

# ─── Paths / env (same on both nodes) ──────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3.yaml}"

export REPO_ROOT
export SLIME_DIR="${REPO_ROOT}/slime"
export MEGATRON_DIR="${MEGATRON_DIR:-${REPO_ROOT}/Megatron-LM}"

source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

HF_HOME="${HF_HOME:-}"
HF_CKPT="${HF_CKPT:-}"
REF_LOAD="${REF_LOAD:-}"
SAVE_CKPT="${SAVE_CKPT:-}"
RESUME_LOAD="${RESUME_LOAD:-${SAVE_CKPT}}"
ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export MASTER_ADDR="${HEAD_IP}"

# ─── env-pool router (head node only) ──────────────────────────────────────────
export USE_REMOTE_ENV="${USE_REMOTE_ENV:-1}"
export PROVIDER_NAME="${PROVIDER_NAME:-pull}"
export ENV_SERVER_BIND_HOST="${ENV_SERVER_BIND_HOST:-0.0.0.0}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18080}"
export ENV_SERVER_HOST="${ENV_SERVER_HOST:-${HEAD_IP}}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-}"
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-0}"
export WORKER_URLS="${WORKER_URLS:-}"

ROUTER_SESSION_NAME="${ROUTER_SESSION_NAME:-terminal_router}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:?CONDA_ENV_PATH must be set}"
ROUTER_PROJECT_DIR="${ROUTER_PROJECT_DIR:-${REPO_ROOT}}"
export CONDA_ENV_PATH
CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.12}"
export CONDA_PYTHON_VERSION
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-${ENV_SERVER_PORT}}"

CHECK_HOST="${CHECK_HOST:-127.0.0.1}"
CHECK_WAIT_SECS="${CHECK_WAIT_SECS:-60}"

# ─── slime args ────────────────────────────────────────────────────────────────
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --load "${RESUME_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 20
  --rotary-base 1000000
)

ROLLOUT_ARGS=(
   --prompt-data "${ROLLOUT_PROMPT_DATA}"
   --input-key task
   --rollout-shuffle
   --reward-key score
   --num-rollout 2000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-max-context-len 16384
   --rollout-temperature 1

   --num-steps-per-rollout 2
   --balance-data
)

EVAL_ARGS=(
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

# Actor: TP=2, EP=4 → world = TP × PP × CP × DP = 2 × 1 × 1 × 4 = 8 GPUs.
# Megatron auto-derives DP = world / (TP × PP × CP) = 4.
#
# Why TP=2 (NOT 4): Qwen3.5-35B-A3B has --num-query-groups 2 (only 2 KV heads),
# and Megatron requires num_query_groups % TP == 0. TP=4 would fail with
# "num_query_groups (2) must be a multiple of tensor_model_parallel_size (4)".
# TP can only be 1 or 2 on this model.
#
# Megatron sets up EP groups within (DP × TP); we need EP × ETP <= DP × TP.
# Here: 4 × 1 = 4 <= 4 × 2 = 8 ✓.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --dynamic_history
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type k3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group ${WANDB_GROUP}
    --wandb-key ${WANDB_KEY}
    # --use-swanlab-sync
    # --swanlab-mode local
    # --swanlab-logdir ${SWAN_LOG_DIR}
  )
else
  WANDB_ARGS=()
fi

TB_ARGS=()
if [[ -n "${TENSORBOARD_DIR:-}" ]]; then
  export TENSORBOARD_DIR
  TB_ARGS=(
    --use-tensorboard
    --tb-project-name "${TB_PROJECT_NAME:-openclaw-rl-terminal}"
    --tb-experiment-name "${TB_EXPERIMENT_NAME:-$(date +%Y%m%d_%H%M%S)}"
  )
  log "TensorBoard enabled, writing to TENSORBOARD_DIR=${TENSORBOARD_DIR}"
fi

# sglang: 8 GPUs / 2 GPUs-per-engine = 4 engines, each TP=2
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static 0.6
   --sglang-enable-dp-attention
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate.generate
   --custom-rollout-log-function-path rollout_log.rollout_log
   --custom-config-path "${CUSTOM_CONFIG_PATH}"
)

# ─── helpers ───────────────────────────────────────────────────────────────────
cleanup_prev() {
  log "cleanup previous processes on this node"
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 3
  pkill -9 ray || true
  pkill -9 python || true
}

start_router() {
  require_cmd curl
  mkdir -p "${ROUTER_PROJECT_DIR}/logs"
  local logf="${ROUTER_PROJECT_DIR}/logs/router_${ROUTER_PORT}.log"

  "${CONDA_ENV_PATH}/bin/python" -m terminal-rl.router_server \
    --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" --workers "${WORKER_URLS}" \
    > "${logf}" 2>&1 &

  export ROUTER_PID=$!
  log "router started pid=${ROUTER_PID}, log=${logf}"
  sleep 1
  tail -n 50 "${logf}" || true
}

check_router() {
  require_cmd curl
  local base_url="http://${CHECK_HOST}:${ROUTER_PORT}"

  log "wait router healthz up to ${CHECK_WAIT_SECS}s: ${base_url}/healthz"
  for ((i=1; i<=CHECK_WAIT_SECS; i++)); do
    if curl -fsS "${base_url}/healthz" >/dev/null 2>&1; then
      log "router is up"
      break
    fi
    sleep 1
  done
  log "curl ${base_url}/status";  curl -sS "${base_url}/status";  echo
  log "curl ${base_url}/healthz"; curl -sS "${base_url}/healthz"; echo
}

detect_nvlink() {
  local count
  count="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
  if [[ "${count:-0}" -gt 0 ]]; then
    export HAS_NVLINK=1
  else
    export HAS_NVLINK=0
  fi
  log "HAS_NVLINK=${HAS_NVLINK} (detected ${count} NVLink references)"
}

maybe_fill_env_server_url() {
  if [[ "${USE_REMOTE_ENV}" == "1" && -z "${ENV_SERVER_URL}" ]]; then
    export ENV_SERVER_URL="http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}"
    if [[ "${START_ENV_POOL_SERVER}" == "0" ]]; then
      export START_ENV_POOL_SERVER=1
    fi
  fi
  log "ENV_SERVER_URL=${ENV_SERVER_URL} START_ENV_POOL_SERVER=${START_ENV_POOL_SERVER}"
}

start_ray_head() {
  require_cmd ray
  log "[head] start ray head at ${HEAD_IP}:${RAY_HEAD_PORT} (dashboard ${RAY_DASHBOARD_PORT})"
  mkdir -p "${RAY_TMPDIR}"
  ray start --head \
    --node-ip-address "${HEAD_IP}" \
    --port "${RAY_HEAD_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --temp-dir "${RAY_TMPDIR}"
}

start_ray_worker() {
  require_cmd ray
  log "[worker] join ray cluster ${HEAD_IP}:${RAY_HEAD_PORT}"
  mkdir -p "${RAY_TMPDIR}"
  ray start \
    --address "${HEAD_IP}:${RAY_HEAD_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --temp-dir "${RAY_TMPDIR}"
}

wait_for_cluster_gpus() {
  # Wait until ray sees 2 × GPUS_PER_NODE GPUs (i.e. both nodes joined).
  # NOTE: we intentionally disable errexit/pipefail inside this loop because
  # `ray status` may briefly return empty / no resource section right after the
  # head starts, and `grep`'s exit 1 would otherwise kill the whole script under
  # `set -euo pipefail`.
  local need=$((2 * GPUS_PER_NODE))
  log "[head] waiting until ray cluster has ${need} GPUs..."
  local i have status_out
  set +e
  set +o pipefail
  for i in $(seq 1 600); do
    status_out="$(ray status 2>/dev/null)"
    # Strip ANSI color codes, then parse "Usage:  0.0/16.0 GPU" -> 16
    have="$(printf '%s\n' "${status_out}" \
      | sed -E 's/\x1b\[[0-9;]*m//g' \
      | awk '/[0-9]+(\.[0-9]+)?\/[0-9]+(\.[0-9]+)?[[:space:]]+GPU/ {
              n=split($0, a, "/"); sub(/[^0-9.].*$/, "", a[2]); print int(a[2]); exit
            }')"
    have="${have:-0}"
    if [ "${have}" -ge "${need}" ] 2>/dev/null; then
      set -e
      set -o pipefail
      log "[head] ray sees ${have} GPUs (>= ${need})"
      return 0
    fi
    if [ $((i % 10)) -eq 1 ]; then
      log "[head] still waiting... ray sees ${have}/${need} GPUs (attempt ${i}/600)"
    fi
    sleep 2
  done
  set -e
  set -o pipefail
  log "[head] ERROR: timeout waiting for worker node to join"; exit 1
}

build_runtime_env_json() {
  python3 - <<'PY'
import json, os

conda_env = os.environ.get("CONDA_ENV_PATH", "")
py_ver = os.environ.get("CONDA_PYTHON_VERSION", "3.12")
site_packages = f"{conda_env}/lib/python{py_ver}/site-packages" if conda_env else ""

parts = [
  os.environ.get("REPO_ROOT",""),
  os.environ.get("SLIME_PKG_DIR",""),
  os.environ.get("MEGATRON_DIR",""),
  os.environ.get("SCRIPT_DIR",""),
  site_packages,
]
pythonpath = ":".join([p for p in parts if p])

env_vars = {
  "PYTHONPATH": pythonpath,
  "CUDA_DEVICE_MAX_CONNECTIONS": "1",
  "NCCL_NVLS_ENABLE": os.environ.get("HAS_NVLINK","0"),
  "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF",""),
  "USE_REMOTE_ENV": os.environ.get("USE_REMOTE_ENV","0"),
  "ENV_SERVER_URL": os.environ.get("ENV_SERVER_URL",""),
  "TENSORBOARD_DIR": os.environ.get("TENSORBOARD_DIR",""),
}
print(json.dumps({"env_vars": env_vars}))
PY
}

submit_job() {
  log "[head] submit ray job (actor=${ACTOR_NUM_NODES}x${ACTOR_GPUS_PER_NODE} TP=2 EP=4 DP=4, rollout=${ROLLOUT_NUM_GPUS} GPUs / ${ROLLOUT_NUM_GPUS_PER_ENGINE} per engine = $((ROLLOUT_NUM_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE)) engines)"
  local runtime_env_json
  runtime_env_json="$(build_runtime_env_json)"

  ray job submit --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${runtime_env_json}" \
    -- python3 ${SLIME_DIR}/train_async.py \
    --actor-num-nodes ${ACTOR_NUM_NODES} \
    --actor-num-gpus-per-node ${ACTOR_GPUS_PER_NODE} \
    --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${TB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
}

# ─── main ──────────────────────────────────────────────────────────────────────
cleanup_prev
detect_nvlink
export SCRIPT_DIR

case "${NODE_ROLE}" in
  head)
    # env pool router only on head
    start_router
    check_router
    maybe_fill_env_server_url

    start_ray_head
    wait_for_cluster_gpus
    submit_job
    ;;

  worker)
    # workers just join the ray cluster and stay alive
    start_ray_worker
    log "[worker] joined cluster, sleeping forever (Ctrl-C to leave)"
    # Keep the script alive so ray worker process isn't reaped
    tail -f /dev/null
    ;;

  *)
    echo "NODE_ROLE must be 'head' or 'worker', got: ${NODE_ROLE}"
    exit 1
    ;;
esac

# ─── HOW TO LAUNCH ─────────────────────────────────────────────────────────────
#
# Prereqs (both nodes):
#   - Same code (REPO_ROOT), same conda env (CONDA_ENV_PATH), same model files.
#   - HEAD_IP reachable from both nodes; firewall open for
#       ${RAY_HEAD_PORT}      (ray GCS)
#       ${RAY_DASHBOARD_PORT} (ray dashboard / job submission)
#       ${ENV_SERVER_PORT}    (env-pool router, only on head)
#       and the standard ray worker / runtime ports (default 10001-10010, 30000-40000)
#
# ORDER MATTERS — start HEAD first, THEN worker.
#
# Why: this script's `cleanup_prev` does `ray stop --force`. If you start the
# worker first and it attaches to an old ray instance on the head node (or
# stale state), then later `bash ... NODE_ROLE=head` will tear that cluster
# down and bring up a brand-new GCS — the worker is left attached to a dead
# cluster and head's `ray status` will only ever show 8 GPUs.
#
# Always: `mkdir -p` the log directory first (tee won't create parents).
#
# Step 1 — on HEAD node (actor side, here 10.252.199.11):
#
#   mkdir -p /data1/logs/openclaw-rl/terminal-rl
#   HEAD_IP=10.252.199.11 \
#   NODE_ROLE=head \
#   CONDA_ENV_PATH=/data1/env/openclaw-rl \
#   HF_CKPT=/data1/models/Qwen/Qwen3.5-35B-A3B \
#   REF_LOAD=/data1/models/Qwen/Qwen3.5-35B-A3B_torch_dist \
#   SAVE_CKPT=/data3/models/Qwen3.5-35B-A3B-openclaw-terminal-rl-2node/ \
#   ROLLOUT_PROMPT_DATA=/data1/codebase/OpenClaw-RL/terminal-rl/dataset/seta_env_convert/train.jsonl \
#   WORKER_URLS=http://10.254.97.36:80 \
#   bash terminal-rl/terminal-rl_qwen3.5-35b-2node.sh 2>&1 \
#     | tee -a "/data1/logs/openclaw-rl/terminal-rl/run_$(date +%F_%H%M%S)_35B_head.log"
#
#   Wait until you see:
#     [head] start ray head at 10.252.199.11:6379 (dashboard 8265)
#     [head] waiting until ray cluster has 16 GPUs...
#   Now start the worker.
#
# Step 2 — on WORKER node (rollout/sglang side, here 10.252.199.27):
#
#   mkdir -p /data1/logs/openclaw-rl/terminal-rl
#   HEAD_IP=10.252.199.11 \
#   NODE_ROLE=worker \
#   CONDA_ENV_PATH=/data1/env/openclaw-rl \
#   bash terminal-rl/terminal-rl_qwen3.5-35b-2node.sh 2>&1 \
#     | tee -a "/data1/logs/openclaw-rl/terminal-rl/run_$(date +%F_%H%M%S)_35B_worker.log"
#
# Step 3 — verify cluster (from EITHER node):
#   ray status
#   # Expected:  Active: 2 nodes  ;  Usage: 0.0/16.0 GPU
#   # If you only see 8.0 GPU after worker reports "Ray runtime started", the
#   # nodes are in different clusters — see Troubleshooting below.
#
# Stop (do worker FIRST so head doesn't tear it down again):
#   On WORKER:  ray stop --force ;  pkill -9 sglang ;  pkill -9 python
#   On HEAD  :  ray stop --force ;  pkill -9 sglang ;  pkill -9 python
#
# Troubleshooting "head only sees 8 GPUs":
#   - On worker run `ray status`. If it shows just the worker, the worker is in
#     a dead cluster (most likely caused by starting worker before head, or by
#     restarting head while worker was up). Fix:
#         worker$ ray stop --force
#         head$   ray stop --force ;  re-run the head command
#         (wait for "waiting until ray cluster has 16 GPUs")
#         worker$ re-run the worker command
#   - Check L4 reachability:   worker$ nc -vz $HEAD_IP 6379    (must succeed)
#   - Check ray version parity: ray --version    (must match on both nodes)
#   - Behind a firewall, pin ray's worker port range on BOTH nodes:
#         ray start ... --min-worker-port 30000 --max-worker-port 30050
#     and open that range in the firewall.
#
# Once head logs "ray sees 16 GPUs (>= 16)" it will `ray job submit` the slime
# job; slime/ray then places the actor group (8 GPUs, TP=4 EP=4 DP=2) and 4
# sglang engines (each TP=2) on the two nodes — with `--actor-num-gpus-per-node 8`
# the actor takes one whole node and sglang naturally lands on the other.
