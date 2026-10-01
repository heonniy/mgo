#!/usr/bin/env bash
# Real-archer 235B coslot stress run — tight cap forces eviction + staging;
# controller Phase-4 verify (drift) ON every layer; many rounds for race
# reproducibility.  See scripts/m4_stress_235b.py for the checks.
set -uo pipefail

ARCH_DIR=/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot
EP_DIR=/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-coslot
cd "$EP_DIR"

# coslot-only PYTHONPATH (preflight in sweep_common refuses non-coslot repos).
export PYTHONPATH="$ARCH_DIR:$EP_DIR"
export MOE_EP_VERIFY_LAYER_END=1          # drift check every layer (the oracle)
export STRESS_ROUNDS="${STRESS_ROUNDS:-4}"
export STRESS_TOKENS="${STRESS_TOKENS:-12}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
CONFIG="${1:-configs/qwen3_235b_stress.yaml}"
NPROC="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

echo "=== preflight cleanup $(date -u +%FT%TZ) ==="
bash scripts/cleanup_prev_run.sh || echo "[warn] cleanup returned nonzero (continuing)"

echo "=== launch stress: cap-cfg=$CONFIG nproc=$NPROC rounds=$STRESS_ROUNDS tokens=$STRESS_TOKENS ==="
torchrun \
    --standalone --nproc_per_node="$NPROC" --no-python \
    scripts/numa_wrap.sh scripts/m4_stress_235b.py "$CONFIG"
rc=$?
echo "=== stress exited rc=$rc $(date -u +%FT%TZ) ==="
exit $rc
