#!/usr/bin/env bash
set -euo pipefail

PORT=8100
INTERVAL=1
NIXL_LOG=""   # path to vLLM log file for NIXL telemetry (opt-in)

usage() {
  echo "Usage: $0 [--port PORT] [--interval SECONDS] [--nixl-log FILE]" >&2
  echo "  --nixl-log FILE  path to vLLM log file for NIXL KV transfer stats" >&2
  echo "                   (requires NIXL_TELEMETRY_ENABLE=1 on the server)" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)      [[ -n "${2:-}" ]] || usage; PORT="$2";     shift 2 ;;
    --interval)  [[ -n "${2:-}" ]] || usage; INTERVAL="$2"; shift 2 ;;
    --nixl-log)  [[ -n "${2:-}" ]] || usage; NIXL_LOG="$2"; shift 2 ;;
    *) usage ;;
  esac
done

URL="http://localhost:${PORT}/metrics"

# ---------------------------------------------------------------------------
# Read scalar metrics from the Prometheus /metrics endpoint.
# Outputs: prompt_tokens gen_tokens running waiting kv_percent requests_complete
# ---------------------------------------------------------------------------
read_metrics () {
  curl -s "$URL" | awk '
  /^vllm:prompt_tokens_total/        {prompt_tokens += $2}
  /^vllm:generation_tokens_total/    {gen_tokens += $2}
  /^vllm:num_requests_running/       {running += $2}
  /^vllm:num_requests_waiting/       {waiting += $2}

  # request_success_total has a finished_reason label; sum all label variants
  /^vllm:request_success_total/      {requests_complete += $2}

  # v1: KV-cache usage (1.0 = 100%)
  /^vllm:kv_cache_usage_perc/        {kv_sum += $2; kv_n += 1; kv_seen = 1}

  # v0 fallback: GPU KV-cache usage (1.0 = 100%)
  (!kv_seen) && /^vllm:gpu_cache_usage_perc/ {kv_sum += $2; kv_n += 1}

  END {
    kv_avg = (kv_n > 0) ? (kv_sum / kv_n) : 0
    # print: prompt_tokens gen_tokens running waiting kv_percent requests_complete
    printf "%.0f %.0f %d %d %.2f %.0f\n",
           prompt_tokens, gen_tokens, running, waiting,
           (100.0 * kv_avg), requests_complete
  }'
}

# ---------------------------------------------------------------------------
# Parse the latest NIXL "Avg xfer time" from the vLLM log file, if provided.
# Returns "-" when unavailable.
# ---------------------------------------------------------------------------
read_nixl_xfer_ms () {
  if [[ -z "$NIXL_LOG" ]]; then
    echo "-"
    return
  fi
  # Lines look like:
  #   INFO ... [metrics.py:98] KV Transfer metrics: ... Avg xfer time (ms)=9.862, ...
  local val
  val=$(grep -oP 'Avg xfer time \(ms\)=\K[0-9.]+' "$NIXL_LOG" 2>/dev/null | tail -1)
  echo "${val:--}"
}

read prev_prompt_tokens prev_gen_tokens prev_running prev_waiting prev_kv prev_requests_complete \
  <<< "$(read_metrics)"
prev_time=$(date +%s)

while true; do
  sleep "$INTERVAL"

  read prompt_tokens gen_tokens running waiting kv_pct requests_complete <<< "$(read_metrics)"
  nixl_xfer_ms=$(read_nixl_xfer_ms)
  now=$(date +%s)

  awk -v now="$now" \
      -v prev="$prev_time" \
      -v prompt_tok="$prompt_tokens" \
      -v prev_prompt_tok="$prev_prompt_tokens" \
      -v gen_tok="$gen_tokens" \
      -v prev_gen_tok="$prev_gen_tokens" \
      -v running="$running" \
      -v waiting="$waiting" \
      -v kv="$kv_pct" \
      -v req_complete="$requests_complete" \
      -v prev_req_complete="$prev_requests_complete" \
      -v nixl_xfer_ms="$nixl_xfer_ms" '
  BEGIN {
      dt = now - prev
      if (dt <= 0) dt = 1

      d_prompt = prompt_tok - prev_prompt_tok
      d_gen    = gen_tok - prev_gen_tok
      d_req    = req_complete - prev_req_complete

      if (d_prompt < 0) d_prompt = 0
      if (d_gen    < 0) d_gen    = 0
      if (d_req    < 0) d_req    = 0

      prefill_tps  = d_prompt / dt
      decode_tps   = d_gen    / dt
      req_per_sec  = d_req    / dt
      total        = running + waiting

      nixl_col = (nixl_xfer_ms == "-") ? "        -" \
                                       : sprintf("%9.3f", nixl_xfer_ms+0)

      printf "%s  decode_TPS=%8.2f  prefill_TPS=%8.2f  running=%5d  waiting=%5d" \
             "  total=%5d  KV=%6.2f%%  req_complete/s=%6.2f  req_total=%10.0f" \
             "  nixl_avg_xfer_ms=%s" \
             "  prefill_total=%12.0f  gen_total=%12.0f\n",
             strftime("%H:%M:%S"),
             decode_tps, prefill_tps,
             running, waiting, total,
             kv,
             req_per_sec, req_complete,
             nixl_col,
             prompt_tok, gen_tok
  }'

  prev_prompt_tokens=$prompt_tokens
  prev_gen_tokens=$gen_tokens
  prev_requests_complete=$requests_complete
  prev_time=$now
done
