#!/usr/bin/env bash
# Milestone-3 multi-rank forward smoke (NPROC=$(nvidia-smi -L | wc -l)).
set -euo pipefail

CONFIG="${1:-configs/qwen3_30b_auto.yaml}"
# Number of *GPUs*, not CPUs — harness may pre-set NPROC=$(nproc) which
# is the CPU core count. Use NUM_GPUS to override explicitly if needed.
NPROC="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Drain rule (persists via this repo tree): reclaim any previous/crashed run's
# pinned NUMA-shm + GPU residue and BLOCK until cgroup.current drops BEFORE
# launching.  Refuses to launch on top of residue — that overlap is the
# run2-style cudaHostRegister OOM (888GB new + stale 888GB > cgroup cap).
# Skip with MOE_SKIP_DRAIN=1 only if you KNOW the box is already clean.
if [ "${MOE_SKIP_DRAIN:-0}" != "1" ] && ! "$REPO_ROOT/scripts/cleanup_prev_run.sh"; then
    echo "[run_m3] FATAL: residue did not clear — aborting launch." >&2
    exit 1
fi

echo "[run_m3] launching $NPROC processes with config=$CONFIG"

# setsid: run torchrun (and all rank children) in their OWN session/process
# group, detached from this script's controlling terminal.  The 888GB
# cudaHostRegister freezes the box enough to drop the interactive SSH session;
# without setsid a non-detached run is in that session's process group and gets
# SIGHUP/SIGKILL'd when it dies (no EXIT_CODE — exactly the run3/4/5 deaths on a
# clean box).  setsid makes the run survive the SSH/session drop.  (run_m4 does
# the same.)  `wait` so this script still blocks until the run finishes.
# nice: run the tree at low CPU prio so the interactive SSH shell keeps CPU
# during the load (no-root, children inherit).  NOTE: we do NOT ionice-idle the
# tree — lowest disk prio (-c2 -n7) was observed to risk AIO read stalls on this
# shared storage (the 30B-style hang), and it doesn't help the real freeze
# source (the 888GB cudaHostRegister memory pin) anyway.  The decisive SSH fixes
# are setsid (run survives a drop) + client-side mosh/keepalive.
nice -n 10 setsid torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    --no-python \
    scripts/numa_wrap.sh scripts/m3_forward.py "$CONFIG" &
wait $!
