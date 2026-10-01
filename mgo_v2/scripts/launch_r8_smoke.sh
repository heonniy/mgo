#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?set MODEL}"
: "${OFFLOAD_DIR:?set OFFLOAD_DIR}"
: "${SIMILARITY:?set SIMILARITY}"
if [[ "${ADMISSION:-hungarian_same_path}" == *path* || "${ADMISSION:-hungarian_same_path}" == *same* || "${ADMISSION:-hungarian_same_path}" == *swap* ]]; then
  : "${AFFINITY:?the selected admission policy requires AFFINITY}"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MOE_EP_DISABLE_ARCHER_EVICT=1
export MGO_V2_NUMA_STRICT="${MGO_V2_NUMA_STRICT:-1}"

args=(--model "$MODEL" --offload-dir "$OFFLOAD_DIR" --similarity "$SIMILARITY"
      --cache-ratio "${CACHE_RATIO:-0.30}" --eviction "${EVICTION:-coverage}"
      --admission "${ADMISSION:-hungarian_same_path}" --max-new-tokens "${MAX_NEW_TOKENS:-4}")
if [[ -n "${AFFINITY:-}" ]]; then args+=(--affinity "$AFFINITY"); fi
torchrun --standalone --nproc_per_node=8 "$ROOT/examples/qwen3_smoke.py" "${args[@]}"
