#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-/tmp/gpu_telemetry_$(hostname)_$(date +%Y%m%d_%H%M%S).csv}"
INTERVAL_SECONDS="${GPU_TELEMETRY_INTERVAL:-1}"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is not available on $(hostname)" >&2
  exit 1
}

mkdir -p "$(dirname "${OUTPUT_PATH}")"

QUERY_FIELDS=(
  timestamp
  index
  uuid
  pci.bus_id
  utilization.gpu
  utilization.memory
  memory.used
  temperature.gpu
  power.draw
  power.limit
  clocks.current.sm
  clocks.current.memory
)

QUERY_HELP="$(nvidia-smi --help-query-gpu)"
for field in \
  clocks_event_reasons.active \
  clocks_event_reasons.sw_power_cap \
  clocks_event_reasons.sw_thermal_slowdown \
  clocks_event_reasons.hw_thermal_slowdown \
  clocks_event_reasons.hw_slowdown \
  clocks_throttle_reasons.active \
  clocks_throttle_reasons.sw_power_cap \
  clocks_throttle_reasons.sw_thermal_slowdown \
  clocks_throttle_reasons.hw_thermal_slowdown \
  clocks_throttle_reasons.hw_slowdown; do
  if grep -Fq "${field}" <<<"${QUERY_HELP}"; then
    QUERY_FIELDS+=("${field}")
  fi
done

QUERY="$(IFS=,; echo "${QUERY_FIELDS[*]}")"
echo "Collecting GPU telemetry every ${INTERVAL_SECONDS}s on $(hostname) -> ${OUTPUT_PATH}" >&2
exec nvidia-smi \
  --query-gpu="${QUERY}" \
  --format=csv,nounits \
  --loop="${INTERVAL_SECONDS}" \
  >"${OUTPUT_PATH}"
