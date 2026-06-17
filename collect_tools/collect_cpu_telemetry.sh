#!/usr/bin/env bash
# Host (CPU-side) memory + pressure/paging state collector.
# Mirrors collect_gpu_telemetry.sh: 1s samples, CSV output, nohup-friendly.
# Timestamp format matches the GPU telemetry CSV (YYYY/MM/DD HH:MM:SS.mmm)
# so the two can be joined directly.
#
# Usage:
#   OUTDIR=/path/to/logs/cpu-telemetry INTERVAL=1 TOPN=15 \
#     COLLECT_SMAPS_ROLLUP=1 \
#     nohup bash collect_cpu_telemetry.sh >/dev/null 2>&1 &
#
# Outputs (under $OUTDIR):
#   host_<hostname>.csv   one row/sec: meminfo + PSI + CPU/vmstat + load
#   proc_<hostname>.csv   top-N processes: RSS/shared/pinned/PSS/context switches
#   numa_<hostname>.csv   one row/sec/node: NUMA-node memory distribution
#   monitor_<hostname>.pid / .log
#
# Stop with:  kill "$(cat $OUTDIR/monitor_<hostname>.pid)"

set -uo pipefail

INTERVAL="${INTERVAL:-1}"
TOPN="${TOPN:-15}"
COLLECT_SMAPS_ROLLUP="${COLLECT_SMAPS_ROLLUP:-1}"
HOST="$(hostname -s 2>/dev/null || hostname)"
OUTDIR="${OUTDIR:-./cpu-telemetry}"
mkdir -p "$OUTDIR"

HOST_CSV="$OUTDIR/host_${HOST}.csv"
PROC_CSV="$OUTDIR/proc_${HOST}.csv"
NUMA_CSV="$OUTDIR/numa_${HOST}.csv"
PIDFILE="$OUTDIR/monitor_${HOST}.pid"
LOGFILE="$OUTDIR/monitor_${HOST}.log"

echo "$$" > "$PIDFILE"
echo "Collecting CPU/host telemetry every ${INTERVAL}s on ${HOST} -> ${HOST_CSV}" | tee "$LOGFILE"

cleanup() {
  rm -f "$PIDFILE"
}

# Remove a stale PID file on normal exit, schema errors, or signals.
trap cleanup EXIT
trap 'echo "stopped $(date)" >> "$LOGFILE"; exit 0' INT TERM

# ---- headers (write once; refuse to append a different schema) ----
HOST_HEADER="timestamp,mem_total_mb,mem_free_mb,mem_available_mb,buffers_mb,cached_mb,cached_excl_shmem_mb,dirty_mb,writeback_mb,anon_mb,mapped_mb,shmem_mb,shmem_hugepages_mb,shmem_pmdmapped_mb,mlocked_mb,unevictable_mb,pagetables_mb,sec_pagetables_mb,kernel_stack_mb,slab_mb,swap_total_mb,swap_free_mb,swap_used_mb,psi_some_avg10,psi_some_avg60,psi_full_avg10,psi_full_avg60,psi_some_total_us,psi_full_total_us,cpu_user_ticks,cpu_nice_ticks,cpu_system_ticks,cpu_idle_ticks,cpu_iowait_ticks,cpu_irq_ticks,cpu_softirq_ticks,cpu_steal_ticks,ctxt_total,processes_total,procs_running,procs_blocked,pgfault,pgmajfault,pgscan_direct,pgsteal_direct,pswpin,pswpout,load1"
PROC_HEADER="timestamp,pid,rss_mb,vmswap_mb,vmlck_mb,vmpin_mb,rss_anon_mb,rss_file_mb,rss_shmem_mb,pss_mb,pss_anon_mb,pss_file_mb,pss_shmem_mb,locked_mb,threads,voluntary_ctxt_switches,nonvoluntary_ctxt_switches,pmem_pct,comm"
NUMA_HEADER="timestamp,node,mem_total_mb,mem_free_mb,mem_used_mb,active_mb,inactive_mb,anon_pages_mb,file_pages_mb,shmem_mb,unevictable_mb"

ensure_header() {
  local file="$1"
  local expected="$2"
  if [ -s "$file" ]; then
    if [ "$(head -n 1 "$file")" != "$expected" ]; then
      echo "Refusing to append a new telemetry schema to $file; use a new OUTDIR." | tee -a "$LOGFILE" >&2
      exit 2
    fi
  else
    printf '%s\n' "$expected" > "$file"
  fi
}

ensure_header "$HOST_CSV" "$HOST_HEADER"
ensure_header "$PROC_CSV" "$PROC_HEADER"
ensure_header "$NUMA_CSV" "$NUMA_HEADER"

now() { date '+%Y/%m/%d %H:%M:%S.%3N'; }

while :; do
  TS="$(now)"

  # ---------- host line ----------
  # /proc/meminfo (kB) -> MB
  awk -v ts="$TS" '
    /^MemTotal:/      {memtotal=$2}
    /^MemFree:/       {memfree=$2}
    /^MemAvailable:/  {memavail=$2}
    /^Buffers:/       {buffers=$2}
    /^Cached:/        {cached=$2}
    /^Dirty:/         {dirty=$2}
    /^Writeback:/     {writeback=$2}
    /^AnonPages:/     {anon=$2}
    /^Mapped:/        {mapped=$2}
    /^Shmem:/         {shmem=$2}
    /^ShmemHugePages:/{shmemhuge=$2}
    /^ShmemPmdMapped:/{shmempmd=$2}
    /^Mlocked:/       {mlocked=$2}
    /^Unevictable:/   {unevictable=$2}
    /^PageTables:/    {pagetables=$2}
    /^SecPageTables:/ {secpagetables=$2}
    /^KernelStack:/   {kernelstack=$2}
    /^Slab:/          {slab=$2}
    /^SwapTotal:/     {swtot=$2}
    /^SwapFree:/      {swfree=$2}
    END{
      mb=1024.0
      cached_excl_shmem=cached-shmem
      if (cached_excl_shmem < 0) cached_excl_shmem=0
      printf "%s,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f",
        ts, memtotal/mb, memfree/mb, memavail/mb, buffers/mb, cached/mb, cached_excl_shmem/mb,
        dirty/mb, writeback/mb, anon/mb, mapped/mb, shmem/mb, shmemhuge/mb,
        shmempmd/mb, mlocked/mb, unevictable/mb, pagetables/mb, secpagetables/mb,
        kernelstack/mb, slab/mb, swtot/mb, swfree/mb, (swtot-swfree)/mb
    }' /proc/meminfo >> "$HOST_CSV"

  # /proc/pressure/memory (PSI). May be absent on older kernels.
  if [ -r /proc/pressure/memory ]; then
    awk '
      /^some/ {for(i=2;i<=NF;i++){split($i,a,"=");p["some_"a[1]]=a[2]}}
      /^full/ {for(i=2;i<=NF;i++){split($i,a,"=");p["full_"a[1]]=a[2]}}
      END{printf ",%s,%s,%s,%s,%s,%s",
        p["some_avg10"],p["some_avg60"],p["full_avg10"],p["full_avg60"],
        p["some_total"],p["full_total"]}' /proc/pressure/memory >> "$HOST_CSV"
  else
    printf ",,,,,," >> "$HOST_CSV"
  fi

  # /proc/stat cumulative CPU scheduler counters. Compute rates/utilization
  # during analysis by differencing adjacent samples.
  awk '
    /^cpu / {user=$2; nice=$3; systicks=$4; idle=$5; iowait=$6; irq=$7; softirq=$8; steal=$9}
    /^ctxt / {ctxt=$2}
    /^processes / {processes=$2}
    /^procs_running / {running=$2}
    /^procs_blocked / {blocked=$2}
    END{printf ",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
      user,nice,systicks,idle,iowait,irq,softirq,steal,ctxt,processes,running,blocked}' \
    /proc/stat >> "$HOST_CSV"

  # /proc/vmstat paging/reclaim counters (cumulative; diff in analysis)
  awk '
    /^pgfault /             {pgfault=$2}
    /^pgmajfault /          {pgmajfault=$2}
    /^pgscan_direct /       {pgscan+=$2}
    /^pgsteal_direct /      {pgsteal+=$2}
    /^pswpin /              {pswpin=$2}
    /^pswpout /             {pswpout=$2}
    END{printf ",%s,%s,%s,%s,%s,%s",
      pgfault,pgmajfault,pgscan,pgsteal,pswpin,pswpout}' /proc/vmstat >> "$HOST_CSV"

  # load1
  awk '{printf ",%s\n",$1}' /proc/loadavg >> "$HOST_CSV"

  # ---------- per-NUMA-node memory ----------
  for node_file in /sys/devices/system/node/node[0-9]*/meminfo; do
    [ -r "$node_file" ] || continue
    node_name="$(basename "$(dirname "$node_file")")"
    awk -v ts="$TS" -v node="${node_name#node}" '
      $3 == "MemTotal:"    {total=$4}
      $3 == "MemFree:"     {free=$4}
      $3 == "MemUsed:"     {used=$4}
      $3 == "Active:"      {active=$4}
      $3 == "Inactive:"    {inactive=$4}
      $3 == "AnonPages:"   {anon=$4}
      $3 == "FilePages:"   {filepages=$4}
      $3 == "Shmem:"       {shmem=$4}
      $3 == "Unevictable:" {unevictable=$4}
      END{mb=1024.0; printf "%s,%s,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
        ts,node,total/mb,free/mb,used/mb,active/mb,inactive/mb,anon/mb,
        filepages/mb,shmem/mb,unevictable/mb}' "$node_file" >> "$NUMA_CSV"
  done

  # ---------- per-process top-N by RSS ----------
  # smaps_rollup provides proportional shared-memory accounting, but it costs
  # more than /proc/<pid>/status and can be disabled for lower probe overhead.
  ps -eo pid=,rss=,pmem=,comm= --sort=-rss 2>/dev/null | head -n "$TOPN" | \
  while read -r pid rss pmem comm; do
    [ -r "/proc/$pid/status" ] || continue
    status_fields="$(awk -v rss="$rss" '
      /^VmSwap:/                       {sw=$2}
      /^VmLck:/                        {lck=$2}
      /^VmPin:/                        {pin=$2}
      /^RssAnon:/                      {anon=$2}
      /^RssFile:/                      {file=$2}
      /^RssShmem:/                     {shmem=$2}
      /^Threads:/                      {threads=$2}
      /^voluntary_ctxt_switches:/      {vctx=$2}
      /^nonvoluntary_ctxt_switches:/   {nvctx=$2}
      END{mb=1024.0; printf "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%s,%s,%s",
        rss/mb,sw/mb,lck/mb,pin/mb,anon/mb,file/mb,shmem/mb,threads,vctx,nvctx}' \
      "/proc/$pid/status")"

    smaps_fields="0,0,0,0,0"
    if [ "$COLLECT_SMAPS_ROLLUP" = "1" ] && [ -r "/proc/$pid/smaps_rollup" ]; then
      smaps_fields="$(awk '
        /^Pss:/       {pss=$2}
        /^Pss_Anon:/  {anon=$2}
        /^Pss_File:/  {file=$2}
        /^Pss_Shmem:/ {shmem=$2}
        /^Locked:/    {locked=$2}
        END{mb=1024.0; printf "%.0f,%.0f,%.0f,%.0f,%.0f",
          pss/mb,anon/mb,file/mb,shmem/mb,locked/mb}' "/proc/$pid/smaps_rollup" 2>/dev/null || true)"
      if [ -z "$smaps_fields" ]; then
        smaps_fields="0,0,0,0,0"
      fi
    fi

    # Reorder status/smaps fields into the documented CSV schema.
    IFS=',' read -r rss_mb swap_mb lck_mb pin_mb anon_mb file_mb shmem_mb threads vctx nvctx \
      <<< "$status_fields"
    IFS=',' read -r pss_mb pss_anon_mb pss_file_mb pss_shmem_mb locked_mb \
      <<< "$smaps_fields"
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "$TS" "$pid" "$rss_mb" "$swap_mb" "$lck_mb" "$pin_mb" "$anon_mb" \
      "$file_mb" "$shmem_mb" "$pss_mb" "$pss_anon_mb" "$pss_file_mb" \
      "$pss_shmem_mb" "$locked_mb" "$threads" "$vctx" "$nvctx" "$pmem" "$comm" \
      >> "$PROC_CSV"
  done

  sleep "$INTERVAL"
done
