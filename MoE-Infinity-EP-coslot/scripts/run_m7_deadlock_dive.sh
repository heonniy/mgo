#!/usr/bin/env bash
# M7 deadlock deep dive — re-run the smoke deadlock with NCCL flight
# recorder + desync debug enabled. PyTorch dumps a JSON trace of every
# enqueued/completed NCCL work item on watchdog timeout, plus a per-rank
# desync analysis showing which collectives are missing on which rank.
#
# Output:
#   /tmp/nccl_dbg/nccl_trace_rank{0,1}_*.json   (flight recorder dumps)
#   /tmp/m7_dead_dive_<ts>.log                  (overall log)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH=/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot:/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-coslot:${PYTHONPATH:-}

mkdir -p /tmp/nccl_dbg

# Flight recorder: record up to 10000 NCCL work items in a ring buffer.
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
# On watchdog timeout, dump the buffer to TORCH_NCCL_DEBUG_INFO_TEMP_FILE.
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="/tmp/nccl_dbg/nccl_trace"
# Cross-rank desync report (which collective is missing on which rank).
export TORCH_NCCL_DESYNC_DEBUG=1
# Shorten timeout to fail faster (default 10 min, set to 90s).
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60
export TORCH_NCCL_COORD_CHECK_MILSEC=1000
# Surface NCCL INFO to capture exact collective call sites.
export NCCL_DEBUG=INFO

ts=$(date +%Y%m%d-%H%M%S)
LOG="/tmp/m7_dead_dive_${ts}.log"
exec > "$LOG" 2>&1

echo "[wrap] M7 deadlock deep dive ts=${ts}"

# Re-use the existing smoke config that reproduces the deadlock.
port=$((29500 + RANDOM % 1000))
torchrun \
    --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:${port}" \
    --nnodes=1 --nproc_per_node=2 --no-python \
    scripts/numa_wrap.sh scripts/m4_long_gen.py \
    configs/_sweep/m5_smoke.yaml \
    --max_new_tokens 2 --run_tag m7_dead_dive || true

echo "[wrap] exit=$?"
echo "---"
echo "[wrap] NCCL flight recorder dumps:"
ls -la /tmp/nccl_dbg/ 2>/dev/null
