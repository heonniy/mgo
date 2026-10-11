#!/bin/bash
# Fully random fetch rank (RANDOM, code 19; i.i.d. uniform per miss, no quota) for the 4 BW-vs-HAQ cells.
Q=/home/hwlee/mgo-results/haq_serial_decode_20261010
DS=/home/hwlee/mgo-results/haq_serial_deepseek_20261010
while pgrep -f "scripts/run_headline_job.py|python run_headline_job.py" >/dev/null; do sleep 10; done
cd /home/hwlee/mgo-haq/mgo_v2
for C in 50 30; do
 L=ds_random_c${C}_v1
 [ -f $DS/$L/result.json ] && continue
 echo "$(date +%T) start $L"
 MGO_RANDOM_SALT=1 MGO_HEADLINE_ROOT=$DS MGO_DS_DECODE_POLICY_SCHEDULE=RANDOM,RANDOM MGO_DS_H2D_SERIAL_ABLATION=1 MGO_DS_PREFILL_POLICY=BR \
 PATH=/home/hwlee/mgo-tools/native-expert-build/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1 NUMBA_CACHE_DIR=/tmp/mgo-haq-deepseek-numba-v1 \
 timeout 5600 /home/hwlee/sub-moe/phase01/.venv/bin/python scripts/run_headline_job.py --workloads /home/hwlee/mgo-results/deepseek_cache_ablation_20261009/WORKLOADS.json --label $L --system main_OURS --worker headline_ours_deepseek_multipolicy_worker.py --cell DeepSeekV2Lite_ShareGPT_R4_C${C}_B16_L512_O64 --timeout 5400 > $DS/$L.driver.log 2>&1
 echo "$(date +%T) done $L rc=$?"
done
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$Q
for C in 50 30; do
 for round in 1 2; do
  L=c${C}_RANDOM_r${round}
  [ -f $Q/$L/result.json ] && continue
  echo "$(date +%T) start $L"
  MGO_RANDOM_SALT=$round timeout 2400 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C${C}_B16_L512_O64 --repeats 3 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy RANDOM > $Q/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
echo RANDOMDONE
