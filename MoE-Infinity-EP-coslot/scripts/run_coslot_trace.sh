#!/usr/bin/env bash
# Single capture run: 235B EP=4, batch=4/rank (=bs16), cap=4 slots/GPU,
# decode=5, with per-layer routing/plan/cache trace dumped to JSONL.
set -euo pipefail

export ARCHER_DIR="/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot"
export EP_DIR="/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-coslot"
# Force coslot-only PYTHONPATH (the login shell injects the original non-coslot
# repos; the coslot-isolation guard in sweep_common would FATAL on them).
export PYTHONPATH="${ARCHER_DIR}:${EP_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
export RESULTS_DIR="${RESULTS_DIR:-/home/work/hyewon.lee/실험/main_exp/results/coslot_trace_${TS}}"
mkdir -p "$RESULTS_DIR"
cd "$EP_DIR"
source "$EP_DIR/scripts/sweep_common.sh"
trap cleanup_on_exit EXIT INT TERM

MODEL=${MODEL:-235b}        # 235b (EP=4, NUMA-shared) | 30b (EP=2, per-rank)
NUM_GPUS=${NUM_GPUS:-4}
CAP=${CAP:-4}
MAX_NEW=${MAX_NEW:-5}
case "$MODEL" in
    235b) BASE_CFG="$EP_DIR/configs/qwen3_235b_auto.yaml"; SHARED=1 ;;
    30b)  BASE_CFG="$EP_DIR/configs/qwen3_30b_auto.yaml";  SHARED=0 ;;
    *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
RUN_TAG="${MODEL}_bsperrank4_cap${CAP}_dec${MAX_NEW}"

# cap=N explicit config (drop sparse_hbm_ratio so the explicit cap wins).
CFG="$RESULTS_DIR/cfg_${MODEL}_cap${CAP}.yaml"
python3 - "$BASE_CFG" "$CAP" "$CFG" <<'PY'
import sys, yaml
src, cap, dst = sys.argv[1:4]
with open(src) as f: c = yaml.safe_load(f)
c.setdefault("offload", {})["cache_capacity_per_rank"] = int(cap)
c["offload"].pop("sparse_hbm_ratio", None)
c.setdefault("policies", {})["fetch_dispatch"] = "naive"
with open(dst, "w") as f: yaml.safe_dump(c, f)
print("wrote", dst)
PY

preflight

export MOE_INFINITY_USE_NUMA_SHARED_HOST=1
export MOE_EP_DISABLE_ARCHER_EVICT=1
export MOE_EP_VERIFY_LAYER_END=1
export MOE_EP_EVICTION_POLICY=lru
export MOE_EP_ROUTING_TRACE=1
export MOE_EP_ROUTING_TRACE_PATH="$RESULTS_DIR"
export MOE_EP_RUN_TAG="$RUN_TAG"
unset MOE_EP_HEAVY_DEBUG || true

port=$((30000 + RANDOM % 30000))
echo "[trace] RESULTS_DIR=$RESULTS_DIR cap=$CAP bs/rank=4 decode=$MAX_NEW port=$port"
set +e
torchrun \
    --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:${port}" \
    --nnodes=1 --nproc_per_node="$NUM_GPUS" --no-python \
    "$EP_DIR/scripts/numa_wrap.sh" "$EP_DIR/scripts/coslot_routing_trace.py" \
    "$CFG" --max_new_tokens "$MAX_NEW" --run_tag "$RUN_TAG" \
    2>&1 | tee "$RESULTS_DIR/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "[trace] exit=$rc  files:"
ls -la "$RESULTS_DIR"/${RUN_TAG}_* 2>/dev/null || true
echo "$RESULTS_DIR" > /tmp/coslot_trace_dir.txt

# ----- post-run residue drain + verify (사용자 요청: 매 실행후 residue 없게) -----
echo "=== post-run drain $(date -u +%FT%TZ) ==="
pgrep -f 'torchrun|coslot_routing_trace\.py|numa_wrap' | xargs -r kill -9 2>/dev/null || true
for f in /tmp/moe_numa*.sock /tmp/*.init.lock; do [[ -e "$f" ]] && rm -f "$f"; done
sync || true
sleep 10   # let the kernel reclaim the unmapped 444GB memfd (residue window)
if [[ -r /sys/fs/cgroup/memory.current ]]; then
    awk '{printf "[drain] cgroup memory.current=%.1fGB (≈0 이면 residue 없음)\n",$1/1073741824}' \
        /sys/fs/cgroup/memory.current
fi
echo "[drain] leftover procs: $(pgrep -f 'torchrun|coslot_routing_trace\.py' | wc -l)  sockets: $(ls /tmp/moe_numa*.sock 2>/dev/null | wc -l)"
echo "[drain] done"
