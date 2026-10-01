#!/usr/bin/env bash
# Validation sweep — 4 runs, Qwen3-235B, EP=4.
#
# Axes:
#   R1: cap=∞ (cache_capacity_per_rank unset → fallback), placement=naive, async fetch — baseline
#   R2: cap=64, placement=naive, async fetch       — cap pressure, naive
#   R3: cap=64, placement=balanced, async fetch    — cap pressure, balanced
#   R4: cap=64, placement=balanced, SYNC fetch     — async overlap measurement (vs R3)
#
# Each run writes:
#   $RESULTS_DIR/$RID/trace.json                  (per-rank counters aggregated)
#   $RESULTS_DIR/$RID/*_routing_r{0..3}.jsonl     (per-rank routing log)
#   $RESULTS_DIR/$RID/simulation.json             (reference planner output)
#   $RESULTS_DIR/$RID/compare.log                 (diff report)
#   $RESULTS_DIR/$RID/run.log                     (stdout/stderr)
#   $RESULTS_DIR/$RID/moe_ep_a2a_rank*.log        (NCCL collective dump)
#
# Usage:
#   bash scripts/run_validation_sweep.sh [--max-new=N] [--cap-stress=N]
set -euo pipefail

# ----- paths -----------------------------------------------------------------
export ARCHER_DIR="/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot"
export EP_DIR="/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-coslot"
export RESULTS_DIR="${RESULTS_DIR:-/home/work/hyewon.lee/실험/main_exp/results/validation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"
echo "[sweep] results -> $RESULTS_DIR"

cd "$EP_DIR"
source "$EP_DIR/scripts/sweep_common.sh"

# Trap any exit (success/crash/Ctrl-C) → unconditional cleanup.
trap cleanup_on_exit EXIT INT TERM

# ----- args ------------------------------------------------------------------
MAX_NEW=${MAX_NEW:-8}
# 30B has 6144 slots (48L × 128E), per-rank top_k avg 2/layer at EP=4.
# cap=16 gives meaningful eviction pressure without being so small that
# the protected-hits invariant always trips OVER_CAP_FATAL.
CAP_STRESS=${CAP_STRESS:-16}
NUM_GPUS=${NUM_GPUS:-4}
MODEL=${MODEL:-30b}   # 30b | 235b
for arg in "$@"; do
    case "$arg" in
        --max-new=*)    MAX_NEW="${arg#*=}";;
        --cap-stress=*) CAP_STRESS="${arg#*=}";;
        --num-gpus=*)   NUM_GPUS="${arg#*=}";;
        --model=*)      MODEL="${arg#*=}";;
        *) echo "unknown arg: $arg"; exit 2;;
    esac
done
case "$MODEL" in
    30b)  CONFIG_BASE="$EP_DIR/configs/qwen3_30b_auto.yaml" ;;
    235b) CONFIG_BASE="$EP_DIR/configs/qwen3_235b_auto.yaml" ;;
    *) echo "unknown model: $MODEL"; exit 2 ;;
esac
echo "[sweep] model=$MODEL config=$CONFIG_BASE cap_stress=$CAP_STRESS gpus=$NUM_GPUS"

# Preflight gate
preflight

# ----- helpers ---------------------------------------------------------------
make_cfg() {
    local cap="$1" placement="$2" out="$3"
    python3 - "$CONFIG_BASE" "$cap" "$placement" "$out" <<'PY'
import sys, yaml
src, cap, placement, dst = sys.argv[1:5]
with open(src) as f: c = yaml.safe_load(f)
c.setdefault("policies", {})["fetch_dispatch"] = placement
# cap=0 means "leave config alone — fallback / sparse_hbm_ratio decides"
if int(cap) > 0:
    c.setdefault("offload", {})["cache_capacity_per_rank"] = int(cap)
else:
    c.setdefault("offload", {}).pop("cache_capacity_per_rank", None)
with open(dst, "w") as f: yaml.safe_dump(c, f)
PY
    echo "$out"
}

run_one() {
    local rid="$1" cfg="$2" force_sync="$3"
    local rdir="$RESULTS_DIR/$rid"
    mkdir -p "$rdir"

    # Sweep-level env — must be exported so torchrun children inherit.
    # 30B fits per-rank (~58GB << 80GB HBM, ~58GB << per-NUMA RAM) so we use
    # per-rank offload (NO shm) for simpler isolation between runs. 235B
    # needs NUMA-shared shm to fit. Auto-pick based on MODEL.
    if [[ "$MODEL" == "235b" ]]; then
        export MOE_INFINITY_USE_NUMA_SHARED_HOST=1
    else
        export MOE_INFINITY_USE_NUMA_SHARED_HOST=0
    fi
    export MOE_EP_DISABLE_ARCHER_EVICT=1
    export MOE_EP_OVER_CAP_FATAL=1
    export MOE_EP_EVICTION_POLICY=lru   # match reference simulator
    export MOE_EP_ROUTING_LOG=1
    export MOE_EP_DUMP_ROUTING_LOG=1
    export MOE_EP_CONTROLLER_DEBUG=1
    export MOE_EP_FORCE_SYNC_FETCH="$force_sync"
    # leave HEAVY_DEBUG off in sweep (too verbose); enable only when triaging.
    unset MOE_EP_HEAVY_DEBUG || true
    unset MOE_EP_DEBUG_A2A || true

    local port=$((30000 + RANDOM % 30000))
    echo "[run $rid] cfg=$cfg force_sync=$force_sync port=$port"
    set +e
    torchrun \
        --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:${port}" \
        --nnodes=1 --nproc_per_node="$NUM_GPUS" --no-python \
        "$EP_DIR/scripts/numa_wrap.sh" "$EP_DIR/scripts/m4_long_gen.py" \
        "$cfg" --max_new_tokens "$MAX_NEW" --run_tag "$rid" \
        2>&1 | tee "$rdir/run.log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "[run $rid] exit=$rc"

    # Move trace + routing_log files into the run's result dir.
    # gather_and_dump writes to cfg.instrumentation.trace_path (default
    # /tmp/moe_ep_traces). Find the latest file matching this run_tag.
    if compgen -G "/tmp/moe_ep_traces/${rid}_*.json" >/dev/null; then
        mv /tmp/moe_ep_traces/${rid}_*.json /tmp/moe_ep_traces/${rid}_*.csv "$rdir/" 2>/dev/null || true
        mv /tmp/moe_ep_traces/${rid}_*_routing_r*.jsonl "$rdir/" 2>/dev/null || true
    fi

    cleanup_between_runs "$rid"
    return $rc
}

# Cross-check with reference simulator after a run.
verify_one() {
    local rid="$1" cap="$2" placement="$3"
    local rdir="$RESULTS_DIR/$rid"
    local rl_glob="$rdir/${rid}_*_routing_r*.jsonl"
    if ! compgen -G "$rl_glob" >/dev/null; then
        echo "[verify $rid] no routing_log files; skipping cross-check"
        return 0
    fi
    python3 "$EP_DIR/scripts/simulate_cache.py" \
        --routing-glob "$rl_glob" --ep-size "$NUM_GPUS" \
        --num-experts 128 --cap "$cap" --placement "$placement" \
        --out "$rdir/simulation.json" 2>&1 | tee "$rdir/sim.log"
    python3 "$EP_DIR/scripts/compare_trace.py" \
        --simulation "$rdir/simulation.json" \
        --routing-glob "$rl_glob" --strict-rank-dist \
        2>&1 | tee "$rdir/compare.log" || true
}

# ----- matrix ---------------------------------------------------------------
mkdir -p "$EP_DIR/configs/_sweep"
CFG_R1=$(make_cfg 0  naive    "$EP_DIR/configs/_sweep/val_R1_unlimited_naive.yaml")
CFG_R2=$(make_cfg "$CAP_STRESS" naive    "$EP_DIR/configs/_sweep/val_R2_cap${CAP_STRESS}_naive.yaml")
CFG_R3=$(make_cfg "$CAP_STRESS" balanced "$EP_DIR/configs/_sweep/val_R3_cap${CAP_STRESS}_balanced.yaml")
CFG_R4=$(make_cfg "$CAP_STRESS" balanced "$EP_DIR/configs/_sweep/val_R4_cap${CAP_STRESS}_balanced_sync.yaml")

# Use cap=0 means unlimited at the simulator side too.
SIM_R1_CAP=0
SIM_R2_CAP="$CAP_STRESS"
SIM_R3_CAP="$CAP_STRESS"
SIM_R4_CAP="$CAP_STRESS"

run_one R1_unlimited_naive_async    "$CFG_R1" 0 && verify_one R1_unlimited_naive_async    "$SIM_R1_CAP" naive
run_one R2_cap_naive_async          "$CFG_R2" 0 && verify_one R2_cap_naive_async          "$SIM_R2_CAP" naive
run_one R3_cap_balanced_async       "$CFG_R3" 0 && verify_one R3_cap_balanced_async       "$SIM_R3_CAP" balanced
run_one R4_cap_balanced_sync        "$CFG_R4" 1 && verify_one R4_cap_balanced_sync        "$SIM_R4_CAP" balanced

echo
echo "[sweep] DONE → $RESULTS_DIR"
ls -la "$RESULTS_DIR"
