#!/bin/bash
# RANDOM (code 19, i.i.d. uniform fetch rank, no quota) decode, BR prefill; same serial ablation as run_ours.sh.
# Qwen3 and DeepSeek C20/C30/C40, 2 rounds; MGO_RANDOM_SALT = round. Starts after run_ours.sh finishes.
R=/home/hwlee/mgo-results/haq_cache_ablation_20261010
PY=/home/hwlee/sub-moe/phase01/.venv/bin/python
until grep -q ALLDONE $R/run_ours.log; do sleep 20; done
while pgrep -f "run_headline_job.py --workloads" >/dev/null; do sleep 10; done
export MGO_HEADLINE_ROOT=$R
cd /home/hwlee/mgo-haq/mgo_v2/scripts
for spec in "1:30 20 40" "2:40 20 30"; do
 round=${spec%%:*}; ORDER=${spec#*:}
 for C in $ORDER; do
  L=qwen_c${C}_RANDOM_r${round}
  [ -f $R/$L/result.json ] && continue
  [ -d $R/$L ] && mv $R/$L $R/failed_${L}.$(date +%s)
  echo "$(date +%T) start $L"
  MGO_RANDOM_SALT=$round timeout 2400 $PY run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C${C}_B16_L512_O64 --repeats 3 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy RANDOM > $R/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
cd /home/hwlee/mgo-haq/mgo_v2
for spec in "1:30 20 40" "2:40 20 30"; do
 round=${spec%%:*}; ORDER=${spec#*:}
 for C in $ORDER; do
  L=ds_c${C}_RANDOM_r${round}
  [ -f $R/$L/result.json ] && continue
  [ -d $R/$L ] && mv $R/$L $R/failed_${L}.$(date +%s)
  echo "$(date +%T) start $L"
  MGO_RANDOM_SALT=$round MGO_DS_DECODE_POLICY_SCHEDULE=RANDOM,RANDOM MGO_DS_H2D_SERIAL_ABLATION=1 MGO_DS_PREFILL_POLICY=BR \
  PATH=/home/hwlee/mgo-tools/native-expert-build/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1 NUMBA_CACHE_DIR=/tmp/mgo-haq-deepseek-numba-v1 \
  timeout 5600 $PY scripts/run_headline_job.py --workloads /home/hwlee/mgo-results/deepseek_cache_ablation_20261009/WORKLOADS.json --label $L --system main_OURS --worker headline_ours_deepseek_multipolicy_worker.py --cell DeepSeekV2Lite_ShareGPT_R4_C${C}_B16_L512_O64 --timeout 5400 > $R/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
echo ALLDONE
