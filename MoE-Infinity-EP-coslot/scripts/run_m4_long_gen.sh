#!/usr/bin/env bash
# Long-generate sweep: naive vs balanced fetch, prefetch on/off, with bounded
# cache to surface evictions. Each run produces a trace JSON; analyze_trace.py
# summarises and compares.
set -euo pipefail

CONFIG_BASE="${1:-configs/qwen3_30b_auto.yaml}"
NPROC="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MAX_NEW="${MAX_NEW_TOKENS:-32}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p configs/_sweep

# Cap cache so we exercise evictions. Per-rank capacity 200 << 48*128=6144 keys.
make_cfg () {
    local fetch="$1" prefetch="$2" cap="$3"
    local out="configs/_sweep/qwen3_30b_${fetch}_pf${prefetch}_cap${cap}.yaml"
    python3 - "$CONFIG_BASE" "$fetch" "$prefetch" "$cap" "$out" <<'PY'
import sys, yaml
src, fetch, prefetch, cap, dst = sys.argv[1:6]
with open(src) as f: c = yaml.safe_load(f)
c.setdefault("policies", {})["fetch_dispatch"] = fetch
c.setdefault("execution", {})["enable_prefetch"] = (prefetch == "1")
c.setdefault("offload", {})["cache_capacity_per_rank"] = int(cap)
with open(dst, "w") as f: yaml.safe_dump(c, f)
PY
    echo "$out"
}

PIDFILE="${MOE_RUN_PIDFILE:-/tmp/moe_run.pgid}"
export MOE_RUN_PIDFILE="$PIDFILE"

run () {
    local cfg="$1" tag="$2"
    echo "[sweep] $tag — $cfg"
    # Unique master_port per run so the c10d TCPStore from the previous run
    # (which os._exit(0) does not gracefully close) cannot collide.
    local port=$((29500 + RANDOM % 1000))
    # setsid → torchrun becomes its own process-group leader (pgid == pid), so a
    # hung run — including every rank child spawned via numactl — is reapable as
    # one group by cleanup_prev_run.sh. Record the pgid so a crash in THIS run
    # (or a later session) can target exactly these PIDs, never a broad
    # `pkill -f` that could also hit an interactive shell.
    setsid torchrun \
        --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:${port}" \
        --nnodes=1 --nproc_per_node="$NPROC" --no-python \
        scripts/numa_wrap.sh scripts/m4_long_gen.py "$cfg" \
        --max_new_tokens "$MAX_NEW" --run_tag "$tag" &
    local tr_pid=$!
    echo "$tr_pid" > "$PIDFILE"
    local rc=0
    wait "$tr_pid" || rc=$?
    rm -f "$PIDFILE"
    echo "[sweep] $tag exited rc=$rc"

    # (A) Replaces the old blind `sleep 10`: actively reclaim the previous run's
    # pinned / NUMA-shm pages and BLOCK until cgroup.current drops. A clean exit
    # reclaims almost instantly; a crashed run that left a stuck survivor (an
    # NCCL watchdog can hold a shared memfd for ~120min) is killed here. If
    # reclaim can't be confirmed, cleanup_prev_run.sh exits non-zero and we abort
    # rather than launch the next run on top of residue and OOM.
    if ! "$REPO_ROOT/scripts/cleanup_prev_run.sh"; then
        echo "[sweep] FATAL: residue did not clear after '$tag' — aborting." >&2
        exit 1
    fi
    if [ "$rc" -ne 0 ]; then
        echo "[sweep] run '$tag' failed (rc=$rc) — aborting sweep." >&2
        exit "$rc"
    fi
}

CFG_NAIVE_NOPF=$(make_cfg naive 0 200)
CFG_NAIVE_PF=$(make_cfg naive 1 200)
CFG_BAL_NOPF=$(make_cfg balanced 0 200)
CFG_BAL_PF=$(make_cfg balanced 1 200)

# (C) Clear any residue from a previous (possibly crashed) session before the
# first launch — a stuck rank holding a shared memfd would otherwise push this
# sweep straight into an OOM.
if ! "$REPO_ROOT/scripts/cleanup_prev_run.sh"; then
    echo "[sweep] FATAL: pre-sweep residue did not clear — aborting." >&2
    exit 1
fi

run "$CFG_NAIVE_NOPF" "naive_nopf"
run "$CFG_NAIVE_PF"   "naive_pf"
run "$CFG_BAL_NOPF"   "balanced_nopf"
run "$CFG_BAL_PF"     "balanced_pf"

echo
echo "[sweep] done — analysing latest traces"
ls -1t /tmp/moe_ep_traces/*.json | head -4 | xargs python3 scripts/analyze_trace.py
