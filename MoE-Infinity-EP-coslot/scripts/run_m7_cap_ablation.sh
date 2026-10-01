#!/usr/bin/env bash
# M7 cap ablation — sweep cache_capacity_per_rank to map the cap → hit-rate
# → ms/tok curve. Working-set ~ num_layers × num_experts = 48*128 = 6144 keys
# for Qwen3-30B; cap << ws stresses evictions, cap >> ws hits steady-state.
set -euo pipefail

CONFIG_BASE="${1:-configs/qwen3_30b_auto.yaml}"
NPROC="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MAX_NEW="${MAX_NEW_TOKENS:-16}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH=/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot:/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-coslot:${PYTHONPATH:-}

mkdir -p configs/_sweep

make_cfg () {
    local cap="$1"
    local out="configs/_sweep/cap_${cap}.yaml"
    python3 - "$CONFIG_BASE" "$cap" "$out" <<'PY'
import sys, yaml
src, cap, dst = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f: c = yaml.safe_load(f)
c.setdefault("policies", {})["fetch_dispatch"] = "naive"
c.setdefault("execution", {})["enable_prefetch"] = False
c.setdefault("offload", {})["cache_capacity_per_rank"] = int(cap)
with open(dst, "w") as f: yaml.safe_dump(c, f)
PY
    echo "$out"
}

run () {
    local cap="$1"
    local cfg
    cfg="$(make_cfg "$cap")"
    echo "[cap-ablation] cap=$cap"
    local port=$((29500 + RANDOM % 1000))
    torchrun \
        --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:${port}" \
        --nnodes=1 --nproc_per_node="$NPROC" --no-python \
        scripts/numa_wrap.sh scripts/m4_long_gen.py "$cfg" \
        --max_new_tokens "$MAX_NEW" --run_tag "cap${cap}"
    sleep 10
}

# Sweep: from heavy-eviction to full working set + headroom.
for cap in 200 500 1000 2000 5000 8000; do
    run "$cap"
done

echo "[cap-ablation] done — analysing latest traces"
ls -1t /tmp/moe_ep_traces/cap*_naive_pf0_*.json | head -6 | xargs python3 scripts/analyze_trace.py
