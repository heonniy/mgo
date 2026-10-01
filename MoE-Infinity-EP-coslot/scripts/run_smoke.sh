#!/usr/bin/env bash
# Smoke driver — launches one process per visible GPU with NUMA-local binding.
#
# Usage:
#   bash scripts/run_smoke.sh [config.yaml]
#
# Default config: configs/qwen3_30b_auto.yaml (small enough for NUMA replication
# under the local cgroup memory cap).
set -euo pipefail

CONFIG="${1:-configs/qwen3_30b_auto.yaml}"
NPROC="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Drain rule (persists via this repo tree): reclaim any previous/crashed run's
# residue and BLOCK until cgroup drops BEFORE launching; refuse to launch on top
# of residue (the run2-style cudaHostRegister OOM).  MOE_SKIP_DRAIN=1 to skip.
if [ "${MOE_SKIP_DRAIN:-0}" != "1" ] && ! "$REPO_ROOT/scripts/cleanup_prev_run.sh"; then
    echo "[run_smoke] FATAL: residue did not clear — aborting launch." >&2
    exit 1
fi

echo "[run_smoke] launching $NPROC processes with config=$CONFIG"

# `--no-python` makes torchrun invoke our wrapper directly (which then exec's
# `numactl ... python scripts/m1_smoke.py CONFIG`). Without it torchrun would
# prepend `python` and `numa_wrap.sh` would be passed as a script.
torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    --no-python \
    scripts/numa_wrap.sh scripts/m1_smoke.py "$CONFIG"
