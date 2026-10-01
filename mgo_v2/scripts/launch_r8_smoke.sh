#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?set MODEL}"
: "${OFFLOAD_DIR:?set OFFLOAD_DIR}"
: "${SIMILARITY:?set SIMILARITY}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MOE_EP_DISABLE_ARCHER_EVICT=1
export MGO_V2_NUMA_STRICT="${MGO_V2_NUMA_STRICT:-1}"

torchrun --standalone --nproc_per_node=8   "$ROOT/examples/qwen3_smoke.py"   --model "$MODEL"   --offload-dir "$OFFLOAD_DIR"   --similarity "$SIMILARITY"   ${AFFINITY:+--affinity "$AFFINITY"}   --cache-ratio "${CACHE_RATIO:-0.30}"   --eviction "${EVICTION:-coverage}"   --admission "${ADMISSION:-hungarian_same_path}"   --max-new-tokens "${MAX_NEW_TOKENS:-4}"
