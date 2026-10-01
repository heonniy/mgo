#!/usr/bin/env bash
# M7 NCCL tuning sweep — runs M4 long-gen under 3 NCCL configurations to
# compare ms/tok. The 2-GPU H100 NV18 (NVLink full-bandwidth) topology means
# big-message wins are small; the interesting axis is small-message latency
# (demand-vector all_gather, count exchanges).
set -euo pipefail

CONFIG="${1:-configs/qwen3_30b_auto.yaml}"
NPROC="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MAX_NEW="${MAX_NEW_TOKENS:-16}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH=/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot:/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-coslot:${PYTHONPATH:-}

run () {
    local tag="$1"; shift
    echo "[nccl-sweep] $tag"
    local port=$((29500 + RANDOM % 1000))
    # All env-var overrides for this run are passed as `KEY=VAL` args before
    # torchrun via `env -i` style assignment.
    env "$@" \
        torchrun \
        --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:${port}" \
        --nnodes=1 --nproc_per_node="$NPROC" --no-python \
        scripts/numa_wrap.sh scripts/m4_long_gen.py "$CONFIG" \
        --max_new_tokens "$MAX_NEW" --run_tag "nccl_${tag}"
    sleep 10
}

# 1. Baseline — let NCCL choose everything.
run baseline

# 2. Low-latency proto for small msgs (demand vector all_gather is 128 bytes).
run ll \
    NCCL_PROTO=LL \
    NCCL_ALGO=Ring \
    NCCL_P2P_LEVEL=NVL \
    TORCH_NCCL_AVOID_RECORD_STREAMS=1

# 3. Throughput-oriented — bigger NCCL buffers, LL128 mixed.
run ll128 \
    NCCL_PROTO=LL128 \
    NCCL_ALGO=Ring \
    NCCL_P2P_LEVEL=NVL \
    NCCL_BUFFSIZE=8388608 \
    TORCH_NCCL_AVOID_RECORD_STREAMS=1

echo "[nccl-sweep] done — analysing latest traces"
ls -1t /tmp/moe_ep_traces/nccl_*.json | head -3 | xargs python3 scripts/analyze_trace.py
