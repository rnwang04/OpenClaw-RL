#!/usr/bin/env bash
# Collect one-time host/container/GPU/software state before a performance run.
#
# Usage:
#   bash collect_prerun_snapshot.sh /path/to/outdir
#   OUTDIR=/path/to/outdir bash collect_prerun_snapshot.sh
#
# Run this once per node/container before launching the RL job. It is read-only.

set -uo pipefail

HOST="$(hostname -s 2>/dev/null || hostname)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${1:-${OUTDIR:-./prerun_snapshot_${HOST}_${STAMP}}}"

mkdir -p "$OUTDIR"/{commands,proc,sys,cgroup}

LOG="$OUTDIR/collect_prerun_snapshot.log"
exec > >(tee -a "$LOG") 2>&1

echo "snapshot_outdir=$OUTDIR"
echo "timestamp=$(date -Is)"
echo "hostname=$(hostname 2>/dev/null || true)"
echo "pwd=$(pwd)"
echo "user=$(id 2>/dev/null || true)"

run_cmd() {
  local name="$1"
  shift
  local file="$OUTDIR/commands/${name}.txt"
  {
    echo "# command: $*"
    echo "# timestamp: $(date -Is)"
    "$@"
    local rc=$?
    echo
    echo "# exit_code: $rc"
    return 0
  } >"$file" 2>&1
}

copy_if_readable() {
  local src="$1"
  local dst="$2"
  if [ -r "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst" 2>/dev/null || true
  fi
}

dump_cgroup_file() {
  local base="$1"
  local rel="$2"
  if [ -r "$base/$rel" ]; then
    mkdir -p "$OUTDIR/cgroup/$(dirname "$rel")"
    cp "$base/$rel" "$OUTDIR/cgroup/$rel" 2>/dev/null || true
  fi
}

# Basic OS/container state.
run_cmd uname_a uname -a
run_cmd os_release sh -c 'cat /etc/os-release 2>/dev/null || true; lsb_release -a 2>/dev/null || true'
run_cmd uptime uptime
run_cmd free_h free -h
run_cmd df_hT df -hT
run_cmd df_dev_shm df -hT /dev/shm
run_cmd mounts sh -c 'findmnt -A 2>/dev/null || mount'
run_cmd ulimit_a bash -lc 'ulimit -a'
run_cmd env_selected sh -c 'env | sort | grep -E "^(CUDA|NCCL|NV|PYTORCH|TORCH|TRITON|RAY|SLIME|MEGATRON|SGLANG|OMP|MKL|UCX|FI_|PATH|LD_LIBRARY_PATH|CONDA|VIRTUAL_ENV|PYTHONPATH)=" || true'

# CPU and NUMA.
run_cmd lscpu lscpu
run_cmd lscpu_e lscpu -e
run_cmd nproc nproc
run_cmd numactl_H sh -c 'numactl -H 2>/dev/null || true'
run_cmd taskset_self sh -c 'taskset -pc $$ 2>/dev/null || true'

copy_if_readable /proc/cpuinfo "$OUTDIR/proc/cpuinfo.txt"
copy_if_readable /proc/meminfo "$OUTDIR/proc/meminfo.txt"
copy_if_readable /proc/vmstat "$OUTDIR/proc/vmstat.txt"
copy_if_readable /proc/stat "$OUTDIR/proc/stat.txt"
copy_if_readable /proc/loadavg "$OUTDIR/proc/loadavg.txt"
copy_if_readable /proc/pressure/cpu "$OUTDIR/proc/pressure_cpu.txt"
copy_if_readable /proc/pressure/memory "$OUTDIR/proc/pressure_memory.txt"
copy_if_readable /proc/pressure/io "$OUTDIR/proc/pressure_io.txt"
copy_if_readable /proc/driver/nvidia/version "$OUTDIR/proc/nvidia_driver_version.txt"

for node_file in /sys/devices/system/node/node[0-9]*/meminfo; do
  [ -r "$node_file" ] || continue
  node="$(basename "$(dirname "$node_file")")"
  copy_if_readable "$node_file" "$OUTDIR/sys/${node}_meminfo.txt"
done

# cgroup v2 state. In containers, /proc/self/cgroup usually maps to a scoped
# path under /sys/fs/cgroup.
copy_if_readable /proc/self/cgroup "$OUTDIR/proc/self_cgroup.txt"
copy_if_readable /proc/self/mountinfo "$OUTDIR/proc/self_mountinfo.txt"
CGROUP_PATH="$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup 2>/dev/null | head -n 1)"
CGROUP_BASE="/sys/fs/cgroup${CGROUP_PATH}"
if [ ! -d "$CGROUP_BASE" ]; then
  CGROUP_BASE="/sys/fs/cgroup"
fi
echo "$CGROUP_BASE" > "$OUTDIR/cgroup/base_path.txt"
for rel in \
  cgroup.controllers cgroup.subtree_control cgroup.type \
  cpu.max cpu.weight cpu.stat cpu.pressure \
  cpuset.cpus cpuset.cpus.effective cpuset.mems cpuset.mems.effective \
  memory.current memory.max memory.high memory.low memory.swap.current memory.swap.max \
  memory.stat memory.numa_stat memory.events memory.pressure \
  pids.current pids.max io.stat; do
  dump_cgroup_file "$CGROUP_BASE" "$rel"
done

# GPU state.
if command -v nvidia-smi >/dev/null 2>&1; then
  run_cmd nvidia_smi_L nvidia-smi -L
  run_cmd nvidia_smi_q nvidia-smi -q
  run_cmd nvidia_smi_topo_m nvidia-smi topo -m
  run_cmd nvidia_smi_query_gpu nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,pstate,power.limit,power.draw,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,memory.total,memory.used,temperature.gpu --format=csv
  run_cmd nvidia_smi_query_compute nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv
  run_cmd nvidia_smi_nvlink sh -c 'nvidia-smi nvlink -s 2>/dev/null || true'
else
  echo "nvidia-smi not found" > "$OUTDIR/commands/nvidia_smi_missing.txt"
fi

# Python/software stack.
run_cmd python_version sh -c 'python3 --version 2>&1 || python --version 2>&1 || true'
run_cmd python_executable sh -c 'python3 - <<PY 2>/dev/null || python - <<PY 2>/dev/null || true
import sys
print(sys.executable)
print(sys.version)
PY'
run_cmd torch_collect_env sh -c 'python3 -m torch.utils.collect_env 2>/dev/null || python -m torch.utils.collect_env 2>/dev/null || true'
run_cmd python_packages sh -c 'python3 - <<PY 2>/dev/null || python - <<PY 2>/dev/null || true
mods = ["torch", "triton", "transformers", "sglang", "ray", "flash_attn", "transformer_engine"]
for name in mods:
    try:
        m = __import__(name)
        print(f"{name}={getattr(m, \"__version__\", \"unknown\")}")
    except Exception as exc:
        print(f"{name}=UNAVAILABLE ({exc})")
PY'
run_cmd pip_freeze sh -c 'python3 -m pip freeze 2>/dev/null || python -m pip freeze 2>/dev/null || true'
run_cmd ray_version sh -c 'ray --version 2>/dev/null || true'

# Repo revisions if run from inside the workspace.
run_cmd git_revisions sh -c '
for d in . slime Megatron-LM terminal-rl openclaw-rl-qizhi; do
  if [ -d "$d/.git" ] || git -C "$d" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "## $d"
    git -C "$d" rev-parse HEAD 2>/dev/null || true
    git -C "$d" status --short 2>/dev/null || true
  fi
done'

echo "done=$(date -Is)"
