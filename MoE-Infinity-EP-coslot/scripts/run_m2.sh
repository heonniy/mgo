#!/usr/bin/env bash
# Milestone-2 forward smoke (single-rank, NPROC=1).
set -euo pipefail

CONFIG="${1:-configs/qwen3_30b_auto.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Drain rule (persists via this repo tree): reclaim any previous/crashed run's
# residue and BLOCK until cgroup drops BEFORE launching; refuse to launch on top
# of residue (the run2-style cudaHostRegister OOM).  MOE_SKIP_DRAIN=1 to skip.
if [ "${MOE_SKIP_DRAIN:-0}" != "1" ] && ! "$REPO_ROOT/scripts/cleanup_prev_run.sh"; then
    echo "[run_m2] FATAL: residue did not clear — aborting launch." >&2
    exit 1
fi

echo "[run_m2] launching 1 process with config=$CONFIG"

torchrun \
    --standalone \
    --nproc_per_node=1 \
    --no-python \
    scripts/numa_wrap.sh scripts/m2_forward.py "$CONFIG"
