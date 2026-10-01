#!/usr/bin/env bash
# Common pre-flight, between-run cleanup, and exit cleanup for sweep scripts.
# Source this from any sweep script:
#     source "$(dirname "${BASH_SOURCE[0]}")/sweep_common.sh"
#
# See feedback_experiment_hygiene memory for the full rationale.

set -euo pipefail

# ----- preflight() -----------------------------------------------------------
preflight() {
    local arch_dir="${ARCHER_DIR:?ARCHER_DIR must be set in caller}"
    local ep_dir="${EP_DIR:?EP_DIR must be set in caller}"
    local store_so="${arch_dir}/moe_infinity/_store.cpython-312-x86_64-linux-gnu.so"
    local engine_so="${arch_dir}/moe_infinity/_engine.cpython-312-x86_64-linux-gnu.so"

    echo "=== preflight $(date -u +%FT%TZ) ==="

    # (1) archer .so present
    if [[ ! -f "$store_so" || ! -f "$engine_so" ]]; then
        echo "[preflight] FATAL archer .so missing — run "
        echo "    cd $arch_dir && CUTLASS_DIR=$HOME/hyewon.lee/cutlass python setup.py build_ext --inplace"
        return 1
    fi
    echo "[preflight] archer .so OK ($(stat -c '%y' "$store_so" | cut -c1-19))"

    # (2) PYTHONPATH isolation
    if [[ -z "${PYTHONPATH:-}" ]]; then
        echo "[preflight] WARN PYTHONPATH empty — exporting fork-only"
    fi
    case ":${PYTHONPATH:-}:" in
        *":${arch_dir}:"*) ;;
        *) export PYTHONPATH="${arch_dir}:${ep_dir}${PYTHONPATH:+:$PYTHONPATH}";;
    esac
    # Refuse to start if upstream MoE-Infinity dir (not the fork) is in PYTHONPATH.
    if echo "$PYTHONPATH" | grep -E '(^|:)/home/work/hyewon\.lee/Baselines-Repository/MoE-Infinity(:|$)' >/dev/null; then
        echo "[preflight] FATAL upstream MoE-Infinity is in PYTHONPATH — fix is not applied to that binary."
        echo "            PYTHONPATH=$PYTHONPATH"
        return 1
    fi
    # coslot isolation (Controller-Owned Slot Execution): refuse the ORIGINAL
    # (non-coslot) EP / archer repos so we never accidentally load the old
    # per-expert staging binary instead of the coslot rewrite.
    if echo "$PYTHONPATH" | grep -E '(^|:)/home/work/hyewon\.lee/Baselines-Repository/MoE-Infinity-EP(-archer)?(:|$)' >/dev/null; then
        echo "[preflight] FATAL original (non-coslot) MoE-Infinity-EP/-archer is in PYTHONPATH."
        echo "            Use MoE-Infinity-EP-coslot / MoE-Infinity-EP-archer-coslot only."
        echo "            PYTHONPATH=$PYTHONPATH"
        return 1
    fi
    echo "[preflight] PYTHONPATH OK (coslot)"

    # (3) zombie torchrun / m*_ processes
    local zombies
    zombies=$(pgrep -af 'torchrun|m[1-4]_(smoke|forward|verify|long_gen)\.py' || true)
    if [[ -n "$zombies" ]]; then
        echo "[preflight] WARN found leftover processes — killing"
        echo "$zombies"
        pgrep -f 'torchrun|m[1-4]_(smoke|forward|verify|long_gen)\.py' \
            | xargs -r kill -9 2>/dev/null || true
        sleep 5
    fi
    echo "[preflight] no leftover torchrun/m*_ processes"

    # (4) stale shm sockets + init locks
    for f in /tmp/moe_numa*.sock /tmp/*.init.lock; do
        [[ -e "$f" ]] && rm -f "$f" && echo "[preflight] unlinked stale $f"
    done

    # (5) GPU memory state
    if command -v nvidia-smi >/dev/null; then
        echo "[preflight] GPU memory state:"
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader \
            | head -8 | sed 's/^/    /'
    fi

    # (6) cgroup
    if [[ -r /sys/fs/cgroup/memory.max && -r /sys/fs/cgroup/memory.current ]]; then
        local cgmax cgcur
        cgmax=$(cat /sys/fs/cgroup/memory.max)
        cgcur=$(cat /sys/fs/cgroup/memory.current)
        echo "[preflight] cgroup memory.current=$cgcur memory.max=$cgmax"
    fi

    echo "[preflight] OK"
    return 0
}

# ----- cleanup_between_runs <run_id> -----------------------------------------
cleanup_between_runs() {
    local rid="${1:-unknown}"
    echo "=== cleanup after $rid ==="
    # Kill any leftover child torchrun processes (os._exit handles success
    # path but not crashes).
    pgrep -f 'torchrun|m[1-4]_(smoke|forward|verify|long_gen)\.py' \
        | xargs -r kill -9 2>/dev/null || true
    # Stale shm sockets / init locks.
    for f in /tmp/moe_numa*.sock /tmp/*.init.lock; do
        [[ -e "$f" ]] && rm -f "$f"
    done
    # Archive a2a per-rank logs into the run's result dir if present.
    if [[ -n "${RESULTS_DIR:-}" && -d "$RESULTS_DIR" ]]; then
        local rdir="$RESULTS_DIR/$rid"
        mkdir -p "$rdir"
        for f in /tmp/moe_ep_a2a_rank*.log; do
            [[ -e "$f" ]] && mv "$f" "$rdir/" 2>/dev/null || true
        done
    fi
    # Unset MOE env we set per run (defensive — sweep sets them again next run).
    unset MOE_EP_FORCE_SYNC_FETCH || true
    unset MOE_EP_OVER_CAP_FATAL || true
    unset MOE_EP_HEAVY_DEBUG || true
    unset MOE_EP_CONTROLLER_DEBUG || true
    unset MOE_EP_DEBUG_A2A || true
    unset MOE_EP_ROUTING_LOG || true
    unset MOE_EP_DUMP_ROUTING_LOG || true
    unset MOE_EP_EVICTION_POLICY || true
    # Brief breather: TCPStore TIME_WAIT, NCCL communicator release.
    sleep 10
}

# ----- trap-installed cleanup at sweep exit ---------------------------------
cleanup_on_exit() {
    echo "=== cleanup_on_exit $(date -u +%FT%TZ) ==="
    pgrep -f 'torchrun|m[1-4]_(smoke|forward|verify|long_gen)\.py' \
        | xargs -r kill -9 2>/dev/null || true
    for f in /tmp/moe_numa*.sock /tmp/*.init.lock; do
        [[ -e "$f" ]] && rm -f "$f"
    done
}
