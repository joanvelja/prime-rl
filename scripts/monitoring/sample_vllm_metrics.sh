#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: VLLM_METRICS_ENDPOINTS="http://host:port/metrics ..." sample_vllm_metrics.sh OUTPUT_DIR [INTERVAL_SECONDS] [DURATION_SECONDS]

Writes one Prometheus scrape per endpoint per interval to OUTPUT_DIR/vllm_metrics.tsv.
Endpoint lists are allocation-specific, so this script deliberately has no hardcoded node defaults.
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

if [[ -z "${VLLM_METRICS_ENDPOINTS:-}" ]]; then
  usage
  exit 2
fi

output_dir=$1
interval=${2:-30}
duration=${3:-0}

mkdir -p "$output_dir"
tsv="$output_dir/vllm_metrics.tsv"

if [[ ! -s "$tsv" ]]; then
  printf 'ts\tendpoint\tmetric\tvalue\n' > "$tsv"
fi

read -r -a endpoints <<< "$VLLM_METRICS_ENDPOINTS"
start_epoch=$(date +%s)

while true; do
  now_epoch=$(date +%s)
  if [[ "$duration" != "0" && $((now_epoch - start_epoch)) -ge "$duration" ]]; then
    exit 0
  fi

  ts=$(date -Iseconds)
  for endpoint in "${endpoints[@]}"; do
    curl -fsS "$endpoint" 2>/dev/null \
      | awk -v ts="$ts" -v endpoint="$endpoint" '
          /^vllm:/ && $0 !~ /^#/ {
            metric=$1
            value=$NF
            gsub(/\t/, " ", metric)
            print ts "\t" endpoint "\t" metric "\t" value
          }
        ' >> "$tsv" || printf '%s\t%s\tscrape_error\t1\n' "$ts" "$endpoint" >> "$tsv"
  done
  sleep "$interval"
done
