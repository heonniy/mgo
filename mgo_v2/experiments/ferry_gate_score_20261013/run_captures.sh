#!/bin/bash
# Serial raw-router-probability captures (GPUs 0,1,4,5), Qwen3 ShareGPT L512, C30.
set -euo pipefail
cd /home/hwlee/mgo-ferry-appendix/mgo_v2
export MGO_HEADLINE_ROOT=/home/hwlee/mgo-results/ferry_gate_score_20261013 MGO_FERRY_WINDOW_PROBS=1
for b in 8 16 64; do
  [ -f $MGO_HEADLINE_ROOT/window_B${b}_L512_H256_v1/trace_receipt.json ] && continue
  echo "=== B$b $(date)"
  /home/hwlee/sub-moe/phase01/.venv/bin/python scripts/run_headline_job.py --label window_B${b}_L512_H256_v1 \
    --system Ours --worker headline_ours_worker.py --cell R4_C30_B${b}_L512_O257 --repeats 1 --policy BR \
    --prefill-optimized --prefill-layout-fast --legacy-decode-layout --capture-eviction-trace \
    --workloads experiments/ferry_gate_score_20261013/workloads/WORKLOADS.json
done
echo "=== DONE $(date)"
