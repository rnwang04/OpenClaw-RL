#!/usr/bin/env bash
# Collect interval telemetry while a run is active.
#
# Usage:
#   OUTDIR=/path/to/runtime_telemetry INTERVAL=1 TOPN=30 \
#     bash collect_runtime_telemetry.sh
#
# Stop:
#   kill "$(cat /path/to/runtime_telemetry/monitor_<host>.pid)"
#
# Outputs:
#   gpu_<host>.csv          nvidia-smi per-GPU samples
#   host_<host>.csv         memory, PSI, CPU/vmstat/load, cgroup counters
#   proc_<host>.csv         top RSS processes plus TARGET_REGEX matches
#   proc_numa_<host>.csv    per-process NUMA page placement summary
#   numa_<host>.csv         per-node memory summary

set -uo pipefail

INTERVAL="${INTERVAL:-1}"
TOPN="${TOPN:-30}"
TARGET_REGEX="${TARGET_REGEX:-MegatronTrainRayActor|SGLangEngine|raylet|plasma|gcs_server|python}"
COLLECT_SMAPS_ROLLUP="${COLLECT_SMAPS_ROLLUP:-1}"
HOST="$(hostname -s 2>/dev/null || hostname)"
OUTDIR="${OUTDIR:-./runtime_telemetry_${HOST}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

HOST_CSV="$OUTDIR/host_${HOST}.csv"
PROC_CSV="$OUTDIR/proc_${HOST}.csv"
PROC_NUMA_CSV="$OUTDIR/proc_numa_${HOST}.csv"
NUMA_CSV="$OUTDIR/numa_${HOST}.csv"
GPU_CSV="$OUTDIR/gpu_${HOST}.csv"
PIDFILE="$OUTDIR/monitor_${HOST}.pid"
LOGFILE="$OUTDIR/monitor_${HOST}.log"

echo "$$" > "$PIDFILE"
echo "Collecting runtime telemetry every ${INTERVAL}s on ${HOST} -> ${OUTDIR}" | tee "$LOGFILE"

cleanup() {
  if [ -n "${GPU_PID:-}" ]; then
    kill "$GPU_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$PIDFILE"
}
trap cleanup EXIT
trap 'echo "stopped $(date -Is)" >> "$LOGFILE"; exit 0' INT TERM

ensure_header() {
  local file="$1"
  local expected="$2"
  if [ -s "$file" ]; then
    if [ "$(head -n 1 "$file")" != "$expected" ]; then
      echo "Refusing to append different schema to $file; use a new OUTDIR." | tee -a "$LOGFILE" >&2
      exit 2
    fi
  else
    printf '%s\n' "$expected" > "$file"
  fi
}

HOST_HEADER="timestamp,mem_total_mb,mem_free_mb,mem_available_mb,cached_mb,cached_excl_shmem_mb,anon_mb,shmem_mb,slab_mb,swap_used_mb,psi_mem_some_avg10,psi_mem_full_avg10,cpu_user_ticks,cpu_system_ticks,cpu_idle_ticks,cpu_iowait_ticks,cpu_steal_ticks,ctxt_total,procs_running,procs_blocked,pgfault,pgmajfault,pgscan_direct,pgsteal_direct,pswpin,pswpout,load1,cg_cpu_usage_usec,cg_cpu_user_usec,cg_cpu_system_usec,cg_nr_throttled,cg_throttled_usec,cg_memory_current_mb,cg_memory_peak_mb,cg_memory_events_oom,cg_memory_events_oom_kill,cg_pids_current"
PROC_HEADER="timestamp,pid,rss_mb,vmswap_mb,vmlck_mb,vmpin_mb,rss_anon_mb,rss_file_mb,rss_shmem_mb,pss_mb,pss_anon_mb,pss_file_mb,pss_shmem_mb,locked_mb,threads,voluntary_ctxt_switches,nonvoluntary_ctxt_switches,cpus_allowed_list,mems_allowed_list,pmem_pct,comm,cmdline"
PROC_NUMA_HEADER="timestamp,pid,node,pages,mb,comm"
NUMA_HEADER="timestamp,node,mem_total_mb,mem_free_mb,mem_used_mb,active_mb,inactive_mb,anon_pages_mb,file_pages_mb,shmem_mb,unevictable_mb"

ensure_header "$HOST_CSV" "$HOST_HEADER"
ensure_header "$PROC_CSV" "$PROC_HEADER"
ensure_header "$PROC_NUMA_CSV" "$PROC_NUMA_HEADER"
ensure_header "$NUMA_CSV" "$NUMA_HEADER"

now() { date '+%Y/%m/%d %H:%M:%S.%3N'; }

CGROUP_PATH="$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup 2>/dev/null | head -n 1)"
CGROUP_BASE="/sys/fs/cgroup${CGROUP_PATH}"
if [ ! -d "$CGROUP_BASE" ]; then
  CGROUP_BASE="/sys/fs/cgroup"
fi
echo "cgroup_base=$CGROUP_BASE" >> "$LOGFILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_FIELDS=(
    timestamp index uuid pci.bus_id utilization.gpu utilization.memory
    memory.used temperature.gpu power.draw power.limit
    clocks.current.sm clocks.current.memory pstate
  )
  QUERY_HELP="$(nvidia-smi --help-query-gpu 2>/dev/null || true)"
  for field in \
    clocks_event_reasons.active \
    clocks_event_reasons.sw_power_cap \
    clocks_event_reasons.hw_thermal_slowdown \
    clocks_event_reasons.hw_slowdown \
    clocks_throttle_reasons.active \
    clocks_throttle_reasons.sw_power_cap \
    clocks_throttle_reasons.hw_thermal_slowdown \
    clocks_throttle_reasons.hw_slowdown; do
    if grep -Fq "$field" <<<"$QUERY_HELP"; then
      GPU_FIELDS+=("$field")
    fi
  done
  GPU_QUERY="$(IFS=,; echo "${GPU_FIELDS[*]}")"
  nvidia-smi --query-gpu="$GPU_QUERY" --format=csv,nounits --loop="$INTERVAL" >"$GPU_CSV" 2>>"$LOGFILE" &
  GPU_PID="$!"
else
  echo "nvidia-smi not found" > "$GPU_CSV"
fi

read_cgroup_value() {
  local rel="$1"
  local default="${2:-}"
  if [ -r "$CGROUP_BASE/$rel" ]; then
    cat "$CGROUP_BASE/$rel" 2>/dev/null
  else
    printf '%s' "$default"
  fi
}

while :; do
  TS="$(now)"

  # Host memory fields.
  awk -v ts="$TS" '
    /^MemTotal:/ {memtotal=$2}
    /^MemFree:/ {memfree=$2}
    /^MemAvailable:/ {memavail=$2}
    /^Cached:/ {cached=$2}
    /^AnonPages:/ {anon=$2}
    /^Shmem:/ {shmem=$2}
    /^Slab:/ {slab=$2}
    /^SwapTotal:/ {swtot=$2}
    /^SwapFree:/ {swfree=$2}
    END {
      mb=1024.0
      cached_excl_shmem=cached-shmem
      if (cached_excl_shmem < 0) cached_excl_shmem=0
      printf "%s,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f",
        ts,memtotal/mb,memfree/mb,memavail/mb,cached/mb,cached_excl_shmem/mb,
        anon/mb,shmem/mb,slab/mb,(swtot-swfree)/mb
    }' /proc/meminfo >> "$HOST_CSV"

  if [ -r /proc/pressure/memory ]; then
    awk '
      /^some/ {for(i=2;i<=NF;i++){split($i,a,"=");p["some_"a[1]]=a[2]}}
      /^full/ {for(i=2;i<=NF;i++){split($i,a,"=");p["full_"a[1]]=a[2]}}
      END{printf ",%s,%s",p["some_avg10"],p["full_avg10"]}' /proc/pressure/memory >> "$HOST_CSV"
  else
    printf ",," >> "$HOST_CSV"
  fi

  awk '
    /^cpu / {user_ticks=$2; sys_ticks=$4; idle=$5; iowait=$6; steal=$9}
    /^ctxt / {ctxt=$2}
    /^procs_running / {running=$2}
    /^procs_blocked / {blocked=$2}
    END{printf ",%s,%s,%s,%s,%s,%s,%s,%s",
      user_ticks,sys_ticks,idle,iowait,steal,ctxt,running,blocked}' /proc/stat >> "$HOST_CSV"

  awk '
    /^pgfault / {pgfault=$2}
    /^pgmajfault / {pgmajfault=$2}
    /^pgscan_direct/ {pgscan+=$2}
    /^pgsteal_direct/ {pgsteal+=$2}
    /^pswpin / {pswpin=$2}
    /^pswpout / {pswpout=$2}
    END{printf ",%s,%s,%s,%s,%s,%s",pgfault,pgmajfault,pgscan,pgsteal,pswpin,pswpout}' /proc/vmstat >> "$HOST_CSV"

  awk '{printf ",%s",$1}' /proc/loadavg >> "$HOST_CSV"

  # cgroup counters, cumulative where applicable.
  cpu_stat="$(read_cgroup_value cpu.stat)"
  cg_cpu_usage="$(awk '$1=="usage_usec"{print $2}' <<<"$cpu_stat")"
  cg_cpu_user="$(awk '$1=="user_usec"{print $2}' <<<"$cpu_stat")"
  cg_cpu_system="$(awk '$1=="system_usec"{print $2}' <<<"$cpu_stat")"
  cg_nr_throttled="$(awk '$1=="nr_throttled"{print $2}' <<<"$cpu_stat")"
  cg_throttled_usec="$(awk '$1=="throttled_usec"{print $2}' <<<"$cpu_stat")"
  mem_events="$(read_cgroup_value memory.events)"
  cg_oom="$(awk '$1=="oom"{print $2}' <<<"$mem_events")"
  cg_oom_kill="$(awk '$1=="oom_kill"{print $2}' <<<"$mem_events")"
  mem_current="$(read_cgroup_value memory.current 0)"
  mem_peak="$(read_cgroup_value memory.peak 0)"
  pids_current="$(read_cgroup_value pids.current 0)"
  printf ",%s,%s,%s,%s,%s,%.0f,%.0f,%s,%s,%s\n" \
    "${cg_cpu_usage:-}" "${cg_cpu_user:-}" "${cg_cpu_system:-}" \
    "${cg_nr_throttled:-}" "${cg_throttled_usec:-}" \
    "$((mem_current / 1024 / 1024))" "$((mem_peak / 1024 / 1024))" \
    "${cg_oom:-}" "${cg_oom_kill:-}" "${pids_current:-}" >> "$HOST_CSV"

  # NUMA-node memory.
  for node_file in /sys/devices/system/node/node[0-9]*/meminfo; do
    [ -r "$node_file" ] || continue
    node_name="$(basename "$(dirname "$node_file")")"
    awk -v ts="$TS" -v node="${node_name#node}" '
      $3 == "MemTotal:" {total=$4}
      $3 == "MemFree:" {free=$4}
      $3 == "MemUsed:" {used=$4}
      $3 == "Active:" {active=$4}
      $3 == "Inactive:" {inactive=$4}
      $3 == "AnonPages:" {anon=$4}
      $3 == "FilePages:" {filepages=$4}
      $3 == "Shmem:" {shmem=$4}
      $3 == "Unevictable:" {unevictable=$4}
      END{mb=1024.0; printf "%s,%s,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
        ts,node,total/mb,free/mb,used/mb,active/mb,inactive/mb,anon/mb,
        filepages/mb,shmem/mb,unevictable/mb}' "$node_file" >> "$NUMA_CSV"
  done

  # Top-N by RSS plus all PIDs matching TARGET_REGEX.
  pid_list="$(
    {
      ps -eo pid=,rss= --sort=-rss 2>/dev/null | head -n "$TOPN" | awk '{print $1}'
      pgrep -f "$TARGET_REGEX" 2>/dev/null || true
    } | awk 'NF && !seen[$1]++ {print $1}'
  )"

  for pid in $pid_list; do
    [ -r "/proc/$pid/status" ] || continue
    comm="$(cat "/proc/$pid/comm" 2>/dev/null | tr ',' ' ' | head -c 80)"
    cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | tr ',' ' ' | head -c 240)"
    pmem="$(ps -p "$pid" -o pmem= 2>/dev/null | awk '{print $1}')"
    status_fields="$(awk '
      /^VmRSS:/ {rss=$2}
      /^VmSwap:/ {sw=$2}
      /^VmLck:/ {lck=$2}
      /^VmPin:/ {pin=$2}
      /^RssAnon:/ {anon=$2}
      /^RssFile:/ {file=$2}
      /^RssShmem:/ {shmem=$2}
      /^Threads:/ {threads=$2}
      /^voluntary_ctxt_switches:/ {vctx=$2}
      /^nonvoluntary_ctxt_switches:/ {nvctx=$2}
      /^Cpus_allowed_list:/ {cpus=$2}
      /^Mems_allowed_list:/ {mems=$2}
      END{mb=1024.0; printf "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%s,%s,%s,%s,%s",
        rss/mb,sw/mb,lck/mb,pin/mb,anon/mb,file/mb,shmem/mb,threads,vctx,nvctx,cpus,mems}' \
      "/proc/$pid/status")"
    smaps_fields="0,0,0,0,0"
    if [ "$COLLECT_SMAPS_ROLLUP" = "1" ] && [ -r "/proc/$pid/smaps_rollup" ]; then
      smaps_fields="$(awk '
        /^Pss:/ {pss=$2}
        /^Pss_Anon:/ {anon=$2}
        /^Pss_File:/ {file=$2}
        /^Pss_Shmem:/ {shmem=$2}
        /^Locked:/ {locked=$2}
        END{mb=1024.0; printf "%.0f,%.0f,%.0f,%.0f,%.0f",pss/mb,anon/mb,file/mb,shmem/mb,locked/mb}' \
        "/proc/$pid/smaps_rollup" 2>/dev/null || true)"
      if [ -z "$smaps_fields" ]; then
        smaps_fields="0,0,0,0,0"
      fi
    fi
    IFS=',' read -r rss_mb swap_mb lck_mb pin_mb anon_mb file_mb shmem_mb threads vctx nvctx cpus mems <<<"$status_fields"
    IFS=',' read -r pss_mb pss_anon_mb pss_file_mb pss_shmem_mb locked_mb <<<"$smaps_fields"
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\"%s\"\n" \
      "$TS" "$pid" "$rss_mb" "$swap_mb" "$lck_mb" "$pin_mb" "$anon_mb" "$file_mb" "$shmem_mb" \
      "$pss_mb" "$pss_anon_mb" "$pss_file_mb" "$pss_shmem_mb" "$locked_mb" "$threads" "$vctx" "$nvctx" \
      "$cpus" "$mems" "${pmem:-}" "$comm" "$cmdline" >> "$PROC_CSV"

    if [ -r "/proc/$pid/numa_maps" ]; then
      awk -v ts="$TS" -v pid="$pid" -v comm="$comm" '
        {
          for (i=1; i<=NF; i++) {
            if ($i ~ /^N[0-9]+=/) {
              split($i, a, "=")
              node=substr(a[1], 2)
              pages[node]+=a[2]
            }
          }
        }
        END {
          for (node in pages) {
            printf "%s,%s,%s,%s,%.1f,%s\n", ts, pid, node, pages[node], pages[node]*4.0/1024.0, comm
          }
        }' "/proc/$pid/numa_maps" >> "$PROC_NUMA_CSV" 2>/dev/null || true
    fi
  done

  sleep "$INTERVAL"
done
