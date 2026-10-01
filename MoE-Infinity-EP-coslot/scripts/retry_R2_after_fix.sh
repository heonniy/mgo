#!/usr/bin/env bash
# Retry R2 (cap=8 naive async) standalone after the archer ExplicitReplaceAsync
# commit-callback fix.  If the fix works, this should NOT deadlock at the
# "All cached expert locked" warning anymore.
#
# Usage: bash scripts/retry_R2_after_fix.sh
set -uo pipefail

ARCHER_DIR=/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot
EP_DIR=/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-coslot
RDIR=/home/work/hyewon.lee/실험/main_exp/results/validation_post_fix_$(date +%H%M%S)

mkdir -p "$RDIR/R2" "$RDIR/R3"

export PYTHONPATH="$ARCHER_DIR:$EP_DIR"
export MOE_INFINITY_USE_NUMA_SHARED_HOST=0
export MOE_EP_DISABLE_ARCHER_EVICT=1
export MOE_EP_OVER_CAP_FATAL=1
export MOE_EP_EVICTION_POLICY=lru
export MOE_EP_ROUTING_LOG=1
export MOE_EP_DUMP_ROUTING_LOG=1
export MOE_EP_FORCE_SYNC_FETCH=0

# Pre-cleanup
pgrep -f 'torchrun|m4_long_gen' | xargs -r kill -9 2>/dev/null || true
rm -f /tmp/moe_numa*.sock /tmp/*.init.lock 2>/dev/null || true
sleep 2

run_one () {
    local rid="$1" cfg="$2"
    local port=$((30000 + RANDOM % 30000))
    echo "[post-fix] === $rid (port=$port) ==="
    cd "$EP_DIR"
    timeout 600 torchrun --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:${port}" \
        --nnodes=1 --nproc_per_node=4 --no-python \
        "$EP_DIR/scripts/numa_wrap.sh" "$EP_DIR/scripts/m4_long_gen.py" \
        "$cfg" --max_new_tokens 8 --run_tag "$rid" \
        > "$RDIR/$rid/run.log" 2>&1
    local rc=$?
    echo "[post-fix] $rid exit=$rc"
    mv /tmp/moe_ep_traces/${rid}_*.{json,csv} "$RDIR/$rid/" 2>/dev/null || true
    mv /tmp/moe_ep_traces/${rid}_*_routing_r*.jsonl "$RDIR/$rid/" 2>/dev/null || true
    # Inter-run cleanup
    pgrep -f 'torchrun|m4_long_gen' | xargs -r kill -9 2>/dev/null || true
    sleep 3
    return $rc
}

# R2 first (the deadlock case)
run_one R2 "$EP_DIR/configs/_sweep/val_R2_cap8_naive.yaml"
R2_rc=$?

# R3 to verify no regression
run_one R3 "$EP_DIR/configs/_sweep/val_R3_cap8_balanced.yaml"
R3_rc=$?

echo ""
echo "=========================================="
echo "[post-fix] R2 (cap=8 naive async) exit=$R2_rc  (was: deadlock)"
echo "[post-fix] R3 (cap=8 balanced async) exit=$R3_rc  (was: PASS @ 512 ms/tok)"
echo "[post-fix] Results: $RDIR"
echo "=========================================="

# Print key signals
for rid in R2 R3; do
    echo ""
    echo "=== $rid signals ==="
    grep -E "ms/tok|generate |drift|fetch_fail|Killed|Traceback|FATAL|All cached expert locked" \
        "$RDIR/$rid/run.log" 2>/dev/null | tail -10
done
