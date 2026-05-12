#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: VLLM_METRICS_ENDPOINTS="host:port host:port ..." sample_vllm_metrics_aggregated.sh OUTPUT_DIR [INTERVAL_SECONDS] [DURATION_SECONDS]

Wide-format RL-oriented companion to sample_vllm_metrics.sh.

Pre-aggregates the per-replica Prometheus scrape into one row per endpoint per
tick with columns useful for RLVR canaries: running/waiting queues, KV cache
pressure (avg + max), finish-reason mix (stop/length/error), preemptions, and
orchestrator-side /pause + /update_weights counters from the vLLM HTTP server.

Endpoint lists are allocation-specific, so this script deliberately has no
hardcoded node defaults.

Output: $OUTPUT_DIR/monitor/vllm_metrics_aggregated.tsv
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

out_dir=$1
interval=${2:-30}
duration=${3:-0}
metrics_dir="$out_dir/monitor"
metrics_file="$metrics_dir/vllm_metrics_aggregated.tsv"
mkdir -p "$metrics_dir"

read -r -a endpoints <<< "$VLLM_METRICS_ENDPOINTS"

if [[ ! -s "$metrics_file" ]]; then
  printf "ts\tepoch\tendpoint\tok\trunning_sum\twaiting_sum\tkv_avg\tkv_max\tpreemptions_sum\tprompt_tokens_sum\tgeneration_tokens_sum\tstop_sum\tlength_sum\terror_sum\tpause_count\tupdate_weights_count\n" > "$metrics_file"
fi

start_epoch=$(date +%s)

while true; do
  ts=$(date -u +%FT%TZ)
  epoch=$(date +%s)
  for endpoint in "${endpoints[@]}"; do
    body=$(curl --max-time 3 -fsS "http://$endpoint/metrics" 2>/dev/null || true)
    if [[ -z "$body" ]]; then
      printf "%s\t%s\t%s\t0\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n" "$ts" "$epoch" "$endpoint" >> "$metrics_file"
      continue
    fi

    awk -v ts="$ts" -v epoch="$epoch" -v endpoint="$endpoint" '
      BEGIN {
        running=0; waiting=0; kv_sum=0; kv_max=0; kv_n=0;
        preemptions=0; prompt_tokens=0; generation_tokens=0;
        stop=0; length_count=0; error=0; pause=0; update_weights=0;
      }
      /^vllm:num_requests_running\{/ { running += $NF }
      /^vllm:num_requests_waiting\{/ { waiting += $NF }
      /^vllm:kv_cache_usage_perc\{/ {
        kv_sum += $NF;
        if ($NF > kv_max) kv_max = $NF;
        kv_n += 1;
      }
      /^vllm:num_preemptions_total\{/ { preemptions += $NF }
      /^vllm:prompt_tokens_total\{/ { prompt_tokens += $NF }
      /^vllm:generation_tokens_total\{/ { generation_tokens += $NF }
      /^vllm:request_success_total\{/ && /finished_reason="stop"/ { stop += $NF }
      /^vllm:request_success_total\{/ && /finished_reason="length"/ { length_count += $NF }
      /^vllm:request_success_total\{/ && /finished_reason="error"/ { error += $NF }
      /^http_requests_total\{/ && /handler="\/pause"/ { pause += $NF }
      /^http_requests_total\{/ && /handler="\/update_weights"/ { update_weights += $NF }
      END {
        kv_avg = kv_n ? kv_sum / kv_n : "NA";
        kv_max_out = kv_n ? kv_max : "NA";
        printf "%s\t%s\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
          ts, epoch, endpoint, running, waiting, kv_avg, kv_max_out,
          preemptions, prompt_tokens, generation_tokens, stop,
          length_count, error, pause, update_weights;
      }
    ' <<< "$body" >> "$metrics_file"
  done

  if (( duration > 0 && epoch - start_epoch >= duration )); then
    break
  fi
  sleep "$interval"
done
