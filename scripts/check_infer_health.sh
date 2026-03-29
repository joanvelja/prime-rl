#!/bin/bash
# Check health of vLLM inference servers for a running GLM5 job.
# Usage: bash scripts/check_infer_health.sh [NUM_INFER_NODES] [POLL_INTERVAL]
# Set POLL_INTERVAL=0 to run once.

NUM_INFER=${1:-12}
POLL_INTERVAL=${2:-5}

PREFILL_PORT=8100
DECODE_PORT=8200
ROUTER_PORT=8000
NUM_PREFILL_NODES=${NUM_PREFILL_NODES:-2}
NUM_PREFILL_REPLICAS=${NUM_PREFILL_REPLICAS:-1}
TOTAL_PREFILL=$((NUM_PREFILL_NODES * NUM_PREFILL_REPLICAS))
NUM_DECODE_NODES=${NUM_DECODE_NODES:-4}
NODES_PER_REPLICA=$((TOTAL_PREFILL + NUM_DECODE_NODES))

# Find the GLM5 job
JOB_LINE=$(squeue -o "%.18i %.30j %N" --noheader 2>/dev/null | grep -i glm5 | head -1)
if [ -z "$JOB_LINE" ]; then
    echo "No running GLM5 job found"
    exit 1
fi

JOB_ID=$(echo "$JOB_LINE" | awk '{print $1}')
NODELIST=$(echo "$JOB_LINE" | awk '{print $3}')
HOSTS=($(scontrol show hostnames "$NODELIST" 2>/dev/null))
INFER_HOSTS=("${HOSTS[@]:0:$NUM_INFER}")
NUM_REPLICAS=$((NUM_INFER / NODES_PER_REPLICA))

check_health() {
    local host=$1 port=$2
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 --max-time 3 "http://${host}:${port}/health" 2>/dev/null)
    [ "$status" = "200" ] && echo "ok" || echo "$status"
}

print_status() {
    local healthy=0 unhealthy=0 total=$((NUM_INFER + NUM_REPLICAS))
    local lines=()

    for ((i=0; i<NUM_INFER; i++)); do
        host="${INFER_HOSTS[$i]}"
        replica=$((i / NODES_PER_REPLICA))
        rank_in_replica=$((i % NODES_PER_REPLICA))
        if [ "$rank_in_replica" -lt "$TOTAL_PREFILL" ]; then
            role="prefill"; port=$PREFILL_PORT
        else
            role="decode"; port=$DECODE_PORT
        fi
        result=$(check_health "$host" "$port")
        if [ "$result" = "ok" ]; then
            label="\033[32m✓\033[0m"
            ((healthy++))
        else
            label="\033[31m✗ ($result)\033[0m"
            ((unhealthy++))
        fi
        lines+=("$(printf "%-32s R%-2d %-7s :%s  " "$host" "$replica" "$role" "$port")$label")
    done

    # Routers
    for ((r=0; r<NUM_REPLICAS; r++)); do
        router_host="${INFER_HOSTS[$((r * NODES_PER_REPLICA))]}"
        result=$(check_health "$router_host" "$ROUTER_PORT")
        if [ "$result" = "ok" ]; then
            label="\033[32m✓\033[0m"
            ((healthy++))
        else
            label="\033[31m✗ ($result)\033[0m"
            ((unhealthy++))
        fi
        lines+=("$(printf "%-32s R%-2d %-7s :%s  " "$router_host" "$r" "router" "$ROUTER_PORT")$label")
    done

    # Clear screen and print
    clear
    echo "Job $JOB_ID | $(date '+%H:%M:%S') | ${healthy}/${total} healthy"
    echo
    for line in "${lines[@]}"; do
        echo -e "$line"
    done
}

if [ "$POLL_INTERVAL" = "0" ]; then
    print_status
else
    while true; do
        print_status
        sleep "$POLL_INTERVAL"
    done
fi
