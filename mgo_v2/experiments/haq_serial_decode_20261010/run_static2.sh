#!/bin/bash
# Qwen STATIC_BLOCK only, 2 rounds x 3 targets per cell (no anchors). Waits for the running DeepSeek job.
Q=/home/hwlee/mgo-results/haq_serial_decode_20261010
while pgrep -f "scripts/run_headline_job.py|python run_headline_job.py" >/dev/null; do sleep 10; done
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$Q
for C in 50 30; do
 for round in 1 2; do
  L=c${C}_STATIC_BLOCK_r${round}
  [ -f $Q/$L/result.json ] && continue
  echo "$(date +%T) start $L"
  timeout 2400 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C${C}_B16_L512_O64 --repeats 3 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy STATIC_BLOCK > $Q/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
echo STATIC2DONE
