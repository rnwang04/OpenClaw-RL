#!/usr/bin/env bash
set -euo pipefail
set -x

# Minimal single-node Qwen3-30B-A3B update_weights benchmark.
#
# Required environment:
#   HF_CKPT=/path/to/Qwen3-30B-A3B
#   REF_LOAD=/path/to/Qwen3-30B-A3B_torch_dist
#   ROLLOUT_PROMPT_DATA=/path/to/train.jsonl
#
# Optional environment:
#   CONDA_ENV_PATH=/path/to/env
#   BENCH_RUN_DIR=/path/to/output
#   NUM_GPUS=8 ACTOR_GPUS=4 ROLLOUT_GPUS=4
#   BENCH_ROLLOUTS=6 START_ROLLOUT_ID=0
#   UPDATE_WEIGHT_BUFFER_SIZE=536870912
#   UPDATE_WEIGHTS_INTERVAL=1
#   UPDATE_WEIGHT_BENCH_PROMPT_LEN=16
#   UPDATE_WEIGHT_BENCH_RESPONSE_LEN=8
#   COLLECT_TELEMETRY=1 CLEANUP_PREV=1

log() { echo "[$(date +'%F %T')] $*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] missing cmd: $1"
    exit 1
  }
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] ${name} is required"
    exit 1
  fi
}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3.yaml}"
COLLECT_TOOLS_DIR="${COLLECT_TOOLS_DIR:-${REPO_ROOT}/collect_tools}"
if [[ ! -f "${COLLECT_TOOLS_DIR}/collect_runtime_telemetry.sh" ]]; then
  COLLECT_TOOLS_DIR="${SCRIPT_DIR}"
fi

export REPO_ROOT
export SCRIPT_DIR
export SLIME_DIR="${REPO_ROOT}/slime"
export MEGATRON_DIR="${MEGATRON_DIR:-${REPO_ROOT}/Megatron-LM}"

if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  if [[ -f "${CONDA_ENV_PATH}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ENV_PATH}/bin/activate"
  fi
  export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -n "${CONDA_ENV_PATH:-}" && -x "${CONDA_ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
fi
export PYTHON_BIN

source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

require_env HF_CKPT
require_env REF_LOAD
require_env ROLLOUT_PROMPT_DATA
require_cmd ray
require_cmd nvidia-smi

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "[ERROR] ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"
BENCH_ROLLOUTS="${BENCH_ROLLOUTS:-6}"
NUM_ROLLOUT="${NUM_ROLLOUT:-$((START_ROLLOUT_ID + BENCH_ROLLOUTS))}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
UPDATE_WEIGHT_BUFFER_SIZE="${UPDATE_WEIGHT_BUFFER_SIZE:-536870912}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-1024}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"

export UPDATE_WEIGHT_BENCH_PROMPT_LEN="${UPDATE_WEIGHT_BENCH_PROMPT_LEN:-16}"
export UPDATE_WEIGHT_BENCH_RESPONSE_LEN="${UPDATE_WEIGHT_BENCH_RESPONSE_LEN:-8}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/openclaw-rl-ray-update-weight-bench}"

# Keep comparable NCCL topology/init logs without logging every collective.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,ENV,NET}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname -s 2>/dev/null || hostname)"
BENCH_RUN_DIR="${BENCH_RUN_DIR:-${REPO_ROOT}/logs/update_weight_bench_${HOST}_${STAMP}}"
mkdir -p "${BENCH_RUN_DIR}"
RUN_LOG="${BENCH_RUN_DIR}/run_${STAMP}_qwen3-30b-update-weight-bench.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

log "run_dir=${BENCH_RUN_DIR}"
log "run_log=${RUN_LOG}"
log "python=${PYTHON_BIN}"

cleanup_runtime_monitor() {
  if [[ -n "${RUNTIME_MONITOR_PID:-}" ]]; then
    kill "${RUNTIME_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_runtime_monitor EXIT

cleanup_prev() {
  log "cleanup previous Ray/SGLang/Python processes"
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 3
  pkill -9 ray || true
  pkill -9 python || true
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

start_ray_head() {
  log "start Ray head"
  mkdir -p "${RAY_TMPDIR}"
  ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
    --temp-dir "${RAY_TMPDIR}"
}

build_runtime_env_json() {
  "${PYTHON_BIN}" - <<'PY'
import json
import os

conda_env = os.environ.get("CONDA_ENV_PATH", "")
py_ver = os.environ.get("CONDA_PYTHON_VERSION", "3.12")
site_packages = f"{conda_env}/lib/python{py_ver}/site-packages" if conda_env else ""

parts = [
    os.environ.get("REPO_ROOT", ""),
    os.environ.get("SLIME_PKG_DIR", ""),
    os.environ.get("MEGATRON_DIR", ""),
    os.environ.get("SCRIPT_DIR", ""),
    site_packages,
]
pythonpath = ":".join([p for p in parts if p])

keys = [
    "PYTHONPATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_NVLS_ENABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "TORCH_NCCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TENSORBOARD_DIR",
    "UPDATE_WEIGHT_BENCH_PROMPT_LEN",
    "UPDATE_WEIGHT_BENCH_RESPONSE_LEN",
]

env_vars = {
    "PYTHONPATH": pythonpath,
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
    "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", os.environ.get("HAS_NVLINK", "0")),
    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    "USE_REMOTE_ENV": "0",
    "ENV_SERVER_URL": "",
}
for key in keys:
    val = os.environ.get(key)
    if val is not None and val != "":
        env_vars[key] = val

print(json.dumps({"env_vars": env_vars}))
PY
}

start_collectors() {
  if [[ "${COLLECT_TELEMETRY:-1}" != "1" ]]; then
    log "telemetry collection disabled"
    return
  fi

  if [[ -f "${COLLECT_TOOLS_DIR}/collect_prerun_snapshot.sh" ]]; then
    bash "${COLLECT_TOOLS_DIR}/collect_prerun_snapshot.sh" "${BENCH_RUN_DIR}/${HOST}-prerun-update-weight-bench"
  else
    log "collect_prerun_snapshot.sh not found under ${COLLECT_TOOLS_DIR}"
  fi

  if [[ -f "${COLLECT_TOOLS_DIR}/collect_runtime_telemetry.sh" ]]; then
    OUTDIR="${BENCH_RUN_DIR}/${HOST}-runtime-update-weight-bench" \
      INTERVAL="${TELEMETRY_INTERVAL:-1}" \
      TOPN="${TELEMETRY_TOPN:-60}" \
      TARGET_REGEX="${TARGET_REGEX:-MegatronTrainRayActor|SGLangEngine|sglang|raylet|plasma|gcs_server|python}" \
      bash "${COLLECT_TOOLS_DIR}/collect_runtime_telemetry.sh" &
    RUNTIME_MONITOR_PID=$!
    echo "${RUNTIME_MONITOR_PID}" > "${BENCH_RUN_DIR}/runtime_monitor.pid"
  else
    log "collect_runtime_telemetry.sh not found under ${COLLECT_TOOLS_DIR}"
  fi
}

submit_job() {
  log "submit Ray job"
  local runtime_env_json
  runtime_env_json="$(build_runtime_env_json)"

  CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${REF_LOAD}"
    --load "${RESUME_LOAD:-${REF_LOAD}}"
    --rotary-base 1000000
  )
  if [[ -n "${SAVE_CKPT:-}" ]]; then
    CKPT_ARGS+=(--save "${SAVE_CKPT}" --save-interval "${SAVE_INTERVAL:-1000000}")
  fi

  ROLLOUT_ARGS=(
    --rollout-function-path update_weight_bench_rollout.generate_rollout
    --prompt-data "${ROLLOUT_PROMPT_DATA}"
    --input-key task
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --start-rollout-id "${START_ROLLOUT_ID}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${UPDATE_WEIGHT_BENCH_RESPONSE_LEN}"
    --rollout-max-context-len "${BENCH_CONTEXT_LEN:-128}"
    --rollout-temperature 0
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
    --balance-data
    --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
    --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE}"
  )

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
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
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

  WANDB_ARGS=()
  if [[ -n "${WANDB_KEY:-}" ]]; then
    WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-openclaw-rl-terminal}"
      --wandb-group "${WANDB_GROUP:-Qwen3-30B-update-weight-bench}"
      --wandb-key "${WANDB_KEY}"
    )
  fi

  TB_ARGS=()
  if [[ -n "${TENSORBOARD_DIR:-}" ]]; then
    TB_ARGS=(
      --use-tensorboard
      --tb-project-name "${TB_PROJECT_NAME:-openclaw-rl-terminal}"
      --tb-experiment-name "${TB_EXPERIMENT_NAME:-update-weight-bench-${STAMP}}"
    )
  fi

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
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
    --custom-rollout-log-function-path rollout_log.rollout_log
    --custom-config-path "${CUSTOM_CONFIG_PATH}"
  )

  ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}" \
    --runtime-env-json="${runtime_env_json}" \
    -- "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${TB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
}

if [[ "${CLEANUP_PREV:-1}" == "1" ]]; then
  cleanup_prev
fi
detect_nvlink
start_collectors
start_ray_head
submit_job

log "done; analyze with:"
log "  python3 terminal-rl/analyze_update_weight_gap.py --run bench=${BENCH_RUN_DIR} --output ${BENCH_RUN_DIR}/update_weight_gap_report.md"
