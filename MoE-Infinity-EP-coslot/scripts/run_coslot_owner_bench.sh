#!/usr/bin/env bash
# Owner-policy benchmark: Qwen3-235B EP=4, naive vs balanced, cap=902 slots/rank
# (≈30% of 3610-expert budget), bs=8/rank, in=48, out=64, SAME seed → identical
# sample_ids + initial condition.  Two runs in one detached process with a
# residue drain after each (사용자 요청: 매 실행후 residue 없게).
set -euo pipefail

export ARCHER_DIR="/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot"
export EP_DIR="/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-coslot"
export PYTHONPATH="${ARCHER_DIR}:${EP_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
export RESULTS_DIR="${RESULTS_DIR:-/home/work/hyewon.lee/실험/main_exp/results/coslot_ownerbench_${TS}}"
mkdir -p "$RESULTS_DIR"
cd "$EP_DIR"
source "$EP_DIR/scripts/sweep_common.sh"
# Robust residue cleanup (kills full tree incl. orphaned ranks + waits for
# shm/GPU reclaim) on ANY exit/crash/Ctrl-C, and before each run below.
trap 'bash "$EP_DIR/scripts/coslot_cleanup.sh"' EXIT INT TERM

NUM_GPUS=${NUM_GPUS:-4}
CAP=${CAP:-902}          # ~30% of 3610 / 4 ranks
SEED=${SEED:-1234}
echo "$RESULTS_DIR" > /tmp/coslot_ownerbench_dir.txt
echo "[ownerbench] RESULTS_DIR=$RESULTS_DIR cap=$CAP seed=$SEED gpus=$NUM_GPUS"

make_cfg() {  # <policy> <out_yaml>
    python3 - "$EP_DIR/configs/qwen3_235b_auto.yaml" "$CAP" "$1" "$2" <<'PY'
import sys, yaml
src, cap, policy, dst = sys.argv[1:5]
with open(src) as f: c = yaml.safe_load(f)
c.setdefault("offload", {})["cache_capacity_per_rank"] = int(cap)
c["offload"].pop("sparse_hbm_ratio", None)
c.setdefault("policies", {})["fetch_dispatch"] = policy
with open(dst, "w") as f: yaml.safe_dump(c, f)
PY
}

run_one() {  # <policy>
    local policy="$1"
    local cfg="$RESULTS_DIR/cfg_${policy}.yaml"
    local out="$RESULTS_DIR/${policy}.json"
    make_cfg "$policy" "$cfg"
    bash "$EP_DIR/scripts/coslot_cleanup.sh"   # ★ always clean BEFORE each run
    preflight
    export MOE_INFINITY_USE_NUMA_SHARED_HOST=1
    export MOE_EP_DISABLE_ARCHER_EVICT=1
    export MOE_EP_VERIFY_LAYER_END=1
    export MOE_EP_EVICTION_POLICY=lru
    # prefill(총 1536 tok × top8)에서 hot expert가 layer당 >1024 tok 받을 수 있어
    # MoEMLP 버퍼 상한(kMaxTokens)을 키움 (recompile 불필요, env 조절).
    export MOE_INFINITY_MAX_TOKENS=4096
    # 235B 로드 AIO 완료 동기화 lost-wakeup은 archer core/aio 근본 fix로 해결:
    # ArcherAioThread::Run 의 accounting+pending--+notify_all 을 scope-guard 하에
    # mutex_ 안에서(성공/에러/예외 전 경로) 수행 + 글로벌 카운터 + Wait/WaitRequestDone
    # 30s timeout self-heal. 따라서 기본은 C++ default(4/rank)로 로드해 fix를
    # 검증까지 겸한다. MOE_IO_THREADS 를 명시하면 override(=1 폴백 가능).
    if [[ -n "${MOE_IO_THREADS:-}" ]]; then export MOE_IO_THREADS; fi
    unset MOE_EP_HEAVY_DEBUG MOE_EP_ROUTING_TRACE || true
    local port=$((30000 + RANDOM % 30000))
    echo "[ownerbench] === RUN owner=$policy cfg=$cfg port=$port ==="
    set +e
    torchrun --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:${port}" \
        --nnodes=1 --nproc_per_node="$NUM_GPUS" --no-python \
        "$EP_DIR/scripts/numa_wrap.sh" "$EP_DIR/scripts/coslot_owner_bench.py" \
        "$cfg" --owner "$policy" --seed "$SEED" --out "$out" \
        2>&1 | tee "$RESULTS_DIR/run_${policy}.log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "[ownerbench] owner=$policy exit=$rc"
    bash "$EP_DIR/scripts/coslot_cleanup.sh"   # ★ always clean AFTER each run
}

run_one naive
run_one balanced

echo "[ownerbench] DONE → $RESULTS_DIR"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || true
