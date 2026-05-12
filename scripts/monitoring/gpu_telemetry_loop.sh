#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OUTPUT_DIR HOST1[,HOST2...] [INTERVAL_SECONDS]" >&2
  exit 2
fi

output_dir=$1
hosts_csv=$2
interval=${3:-30}

mkdir -p "$output_dir"
csv="$output_dir/gpu_telemetry.csv"

if [[ ! -s "$csv" ]]; then
  echo "ts,host,index,name,util_pct,mem_used_mib,mem_total_mib,power_w,temperature_c" > "$csv"
fi

IFS=',' read -r -a hosts <<< "$hosts_csv"

while true; do
  ts=$(date -Iseconds)
  for host in "${hosts[@]}"; do
    [[ -n "$host" ]] || continue
    srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "$host" \
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="$ts" -v host="$host" 'BEGIN { FS=", "; OFS="," } { print ts,host,$0 }' \
      >> "$csv" || true
  done
  sleep "$interval"
done
