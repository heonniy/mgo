#!/usr/bin/env bash
# Milestone-3 automated verification:
#   1) Generate single-rank reference logits for the prompt set (NPROC=1).
#   2) Run multi-rank verification (NPROC=NUM_GPUS), comparing per-rank
#      output against the corresponding reference.
set -euo pipefail

CONFIG="${1:-configs/qwen3_30b_auto.yaml}"
NPROC="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
REF_DIR="${REF_DIR:-/tmp/moe_ep_traces/m2_refs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Per-rank prompts must match scripts/m3_verify.py PROMPTS list.
PROMPTS=(
    "The capital of France is"
    "The largest planet in our solar system is"
    "Photosynthesis converts"
    "The author of 'Pride and Prejudice' is"
    "The chemical symbol for gold is"
    "The speed of light is approximately"
    "The currency of Japan is"
    "The longest river in the world is"
)

mkdir -p "$REF_DIR"

echo "[verify] step 1 — generating single-rank references in $REF_DIR for $NPROC prompts"
for ((i=0; i<NPROC; i++)); do
    PROMPT="${PROMPTS[$i]}"
    SLUG=$(python3 -c "import hashlib,sys; print(hashlib.sha1(sys.argv[1].encode('utf-8')).hexdigest()[:12])" "$PROMPT")
    REF_FILE="$REF_DIR/$SLUG.json"
    if [[ -f "$REF_FILE" ]]; then
        echo "[verify] skip (already cached): $REF_FILE"
        continue
    fi
    echo "[verify] generating ref for prompt: $PROMPT"
    torchrun \
        --standalone --nproc_per_node=1 --no-python \
        scripts/numa_wrap.sh scripts/m2_forward.py "$CONFIG" \
        --prompt "$PROMPT" --max_new_tokens 1 --save_ref "$REF_DIR"
done

echo "[verify] step 2 — running multi-rank verification (NPROC=$NPROC)"
torchrun \
    --standalone --nproc_per_node="$NPROC" --no-python \
    scripts/numa_wrap.sh scripts/m3_verify.py "$CONFIG" --ref_dir "$REF_DIR"
