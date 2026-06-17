#!/usr/bin/env bash
# probe_cpu_throttle.sh
# 目的: 在训练运行期间, 抓取 CPU 满载频率 / governor / cgroup CFS 节流 /
#       NUMA / IRQ / BIOS 电源策略等, 用来定位 bare vs Qizhi 的 host 侧 gap.
#
# 用法:
#   1) 静态快照(起跑前跑一次, 容器内 + bare 各跑一次对比):
#        bash probe_cpu_throttle.sh static  /path/out_dir
#   2) 运行期采样(和训练同时跑, 采到训练结束按 Ctrl-C 或到时长):
#        bash probe_cpu_throttle.sh sample  /path/out_dir  [间隔秒=2] [时长秒=0(0=直到Ctrl-C)]
#   3) 一把梭(先静态再采样):
#        bash probe_cpu_throttle.sh all     /path/out_dir  [间隔秒=2] [时长秒=0]
#
# 重点关注产物:
#   static.txt          —— governor / scaling_driver / cpu.max / cpuset / OMP / numactl / nvidia power
#   cpu_freq.csv        —— 每个采样点的 per-core MHz 统计(满载是否被压在 ~2.1GHz / 75%)
#   cgroup_cpu.csv      —— nr_throttled / throttled_usec 增量(>0 即被 CFS 节流, 实锤)
#   cpu_pressure.csv    —— PSI some/full(CPU 等待压力)
#   turbostat.log       —— 若有 turbostat: 实测频率/功耗/C-state(最权威)

set -uo pipefail

MODE="${1:-all}"
OUT="${2:-./cpu_probe_$(hostname)_$(date +%Y%m%d_%H%M%S)}"
INTERVAL="${3:-2}"
DURATION="${4:-0}"
mkdir -p "$OUT"

# ---- 找到本进程所在的 cgroup v2 路径 ----
detect_cgroup() {
  local rel base
  rel="$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)"
  for base in /sys/fs/cgroup "/sys/fs/cgroup/unified"; do
    if [ -f "${base}${rel}/cpu.stat" ]; then echo "${base}${rel}"; return; fi
    if [ -f "${base}/cpu.stat" ]; then echo "${base}"; return; fi
  done
  echo ""
}
CG="$(detect_cgroup)"

ts() { date +%Y-%m-%dT%H:%M:%S.%3N; }

snapshot_static() {
  local f="$OUT/static.txt"
  {
    echo "# host=$(hostname)  time=$(ts)  cgroup=$CG"
    echo "=== nproc / cpu count ==="
    nproc; echo "online: $(cat /sys/devices/system/cpu/online 2>/dev/null)"
    echo "taskset self: $(taskset -pc $$ 2>/dev/null)"
    echo
    echo "=== cpufreq governor / driver / limits (cpu0..) ==="
    for c in 0 1 32 64 96 128 160 191; do
      d=/sys/devices/system/cpu/cpu$c/cpufreq
      [ -d "$d" ] || continue
      printf "cpu%s gov=%s driver=%s cur=%s min=%s max=%s\n" "$c" \
        "$(cat $d/scaling_governor 2>/dev/null)" \
        "$(cat $d/scaling_driver 2>/dev/null)" \
        "$(cat $d/scaling_cur_freq 2>/dev/null)" \
        "$(cat $d/scaling_min_freq 2>/dev/null)" \
        "$(cat $d/scaling_max_freq 2>/dev/null)"
    done
    echo "intel_pstate status: $(cat /sys/devices/system/cpu/intel_pstate/status 2>/dev/null)"
    echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
    echo "energy_perf_bias (cpu0): $(cat /sys/devices/system/cpu/cpu0/power/energy_perf_bias 2>/dev/null)"
    echo
    echo "=== lscpu (scaling MHz / max / min) ==="
    lscpu 2>/dev/null | grep -iE 'MHz|model name|^CPU\(s\)|NUMA'
    echo
    echo "=== cgroup CPU limits ==="
    echo "cpu.max:    $(cat "$CG/cpu.max" 2>/dev/null)   # '<quota> <period>'; quota/period=可用核数, 'max'=不限"
    echo "cpu.weight: $(cat "$CG/cpu.weight" 2>/dev/null)"
    echo "cpu.max.burst: $(cat "$CG/cpu.max.burst" 2>/dev/null)"
    echo "cpuset.cpus:           $(cat "$CG/cpuset.cpus" 2>/dev/null)"
    echo "cpuset.cpus.effective: $(cat "$CG/cpuset.cpus.effective" 2>/dev/null)"
    echo "cpu.stat:"; sed 's/^/  /' "$CG/cpu.stat" 2>/dev/null
    echo
    echo "=== 关键环境变量(线程池大小) ==="
    env | grep -iE '^(OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|GOMP|TORCH_NUM_THREADS|RAYON)' || echo "(均未设置 -> 多数库默认按 nproc=$(nproc) 开线程, 与 120 核配额不匹配)"
    echo
    echo "=== NUMA 拓扑 ==="
    numactl -H 2>/dev/null || echo "(no numactl)"
    echo
    echo "=== GPU 电源/时钟上限 ==="
    nvidia-smi -q -d POWER,CLOCK 2>/dev/null | grep -iE 'Power Limit|Default|Max Clocks|SM |Memory ' | head -40
    echo
    echo "=== IRQ 分布(前 20 行) ==="
    head -1 /proc/interrupts; grep -iE 'nvidia|mlx|eth|ib' /proc/interrupts 2>/dev/null | head -20
    echo
    echo "=== DMI / BIOS 电源(需 root, 可能无) ==="
    (dmidecode -t processor 2>/dev/null | grep -iE 'Speed|Current|Max') || echo "(no dmidecode)"
  } > "$f" 2>&1
  echo "[static] -> $f"
}

sample_loop() {
  local fcpu="$OUT/cpu_freq.csv" fcg="$OUT/cgroup_cpu.csv" fps="$OUT/cpu_pressure.csv"
  echo "timestamp,n_cpus,mhz_min,mhz_mean,mhz_max" > "$fcpu"
  echo "timestamp,usage_usec,user_usec,system_usec,nr_periods,nr_throttled,throttled_usec" > "$fcg"
  echo "timestamp,some_avg10,some_avg60,full_avg10,full_avg60,some_total,full_total" > "$fps"

  # 后台 turbostat(若有, 最权威的实测频率/功耗/C-state)
  if command -v turbostat >/dev/null 2>&1; then
    ( turbostat --quiet --interval "$INTERVAL" 2>>"$OUT/turbostat.log" >>"$OUT/turbostat.log" ) &
    echo "$!" > "$OUT/turbostat.pid"
    echo "[sample] turbostat 已启动 -> $OUT/turbostat.log"
  else
    echo "(turbostat 不可用; 用 /proc/cpuinfo 的 'cpu MHz' 替代)" > "$OUT/turbostat.log"
  fi

  local start now end
  start=$(date +%s); end=$((start + DURATION))
  echo "[sample] interval=${INTERVAL}s duration=${DURATION}s (0=until Ctrl-C). cgroup=$CG"
  trap '[ -f "$OUT/turbostat.pid" ] && kill "$(cat "$OUT/turbostat.pid")" 2>/dev/null; echo; echo "[sample] stopped"; exit 0' INT TERM

  while :; do
    local T; T="$(ts)"
    # --- per-core MHz from /proc/cpuinfo ---
    awk -v t="$T" '
      /cpu MHz/ { v=$4+0; n++; s+=v; if(min==""||v<min)min=v; if(v>max)max=v }
      END { if(n>0){ printf "%s,%d,%.0f,%.0f,%.0f\n", t, n, min, s/n, max } }
    ' /proc/cpuinfo >> "$fcpu"
    # --- cgroup cpu.stat (累计值, 后处理取增量看 nr_throttled 是否上涨) ---
    if [ -n "$CG" ] && [ -f "$CG/cpu.stat" ]; then
      awk -v t="$T" '
        {v[$1]=$2}
        END{ printf "%s,%s,%s,%s,%s,%s,%s\n", t, v["usage_usec"], v["user_usec"], v["system_usec"], v["nr_periods"], v["nr_throttled"], v["throttled_usec"] }
      ' "$CG/cpu.stat" >> "$fcg"
    fi
    # --- cgroup cpu.pressure (PSI) ---
    if [ -n "$CG" ] && [ -f "$CG/cpu.pressure" ]; then
      awk -v t="$T" '
        /^some/{for(i=2;i<=NF;i++){split($i,p,"="); s[p[1]]=p[2]}}
        /^full/{for(i=2;i<=NF;i++){split($i,p,"="); f[p[1]]=p[2]}}
        END{ printf "%s,%s,%s,%s,%s,%s,%s\n", t, s["avg10"], s["avg60"], f["avg10"], f["avg60"], s["total"], f["total"] }
      ' "$CG/cpu.pressure" >> "$fps"
    fi
    now=$(date +%s)
    [ "$DURATION" -gt 0 ] && [ "$now" -ge "$end" ] && break
    sleep "$INTERVAL"
  done
  [ -f "$OUT/turbostat.pid" ] && kill "$(cat "$OUT/turbostat.pid")" 2>/dev/null
  echo "[sample] done -> $fcpu , $fcg , $fps"
}

case "$MODE" in
  static) snapshot_static ;;
  sample) sample_loop ;;
  all)    snapshot_static; sample_loop ;;
  *) echo "unknown mode: $MODE (use static|sample|all)"; exit 1 ;;
esac
