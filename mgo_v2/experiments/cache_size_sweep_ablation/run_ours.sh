#!/bin/bash
# OURS (HAQ decode, BR prefill), serial GEMM + serial H2D ablation, prefetch off, R4 B16/L512/O64.
# C30 is reused from haq_serial_decode_20261010 / haq_serial_deepseek_20261010.
# Qwen3: 2 rounds x 3 targets (C20,C40 then C40,C20). DeepSeek: 2 jobs x 2 targets per cell. Skips finished labels.
R=/home/hwlee/mgo-results/haq_cache_ablation_20261010
PY=/home/hwlee/sub-moe/phase01/.venv/bin/python
export MGO_HEADLINE_ROOT=$R
cd /home/hwlee/mgo-haq/mgo_v2/scripts
for spec in "1:20 40" "2:40 20"; do
 round=${spec%%:*}; ORDER=${spec#*:}
 for C in $ORDER; do
  L=qwen_c${C}_HAQ_r${round}
  [ -f $R/$L/result.json ] && continue
  [ -d $R/$L ] && mv $R/$L $R/failed_${L}.$(date +%s)
  echo "$(date +%T) start $L"
  timeout 2400 $PY run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C${C}_B16_L512_O64 --repeats 3 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy HAQ > $R/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
cd /home/hwlee/mgo-haq/mgo_v2
for spec in "1:20 40" "2:40 20"; do
 round=${spec%%:*}; ORDER=${spec#*:}
 for C in $ORDER; do
  L=ds_c${C}_HAQ_r${round}
  [ -f $R/$L/result.json ] && continue
  [ -d $R/$L ] && mv $R/$L $R/failed_${L}.$(date +%s)
  echo "$(date +%T) start $L"
  MGO_DS_DECODE_POLICY_SCHEDULE=HAQ,HAQ MGO_DS_H2D_SERIAL_ABLATION=1 MGO_DS_PREFILL_POLICY=BR \
  PATH=/home/hwlee/mgo-tools/native-expert-build/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1 NUMBA_CACHE_DIR=/tmp/mgo-haq-deepseek-numba-v1 \
  timeout 5600 $PY scripts/run_headline_job.py --workloads /home/hwlee/mgo-results/deepseek_cache_ablation_20261009/WORKLOADS.json --label $L --system main_OURS --worker headline_ours_deepseek_multipolicy_worker.py --cell DeepSeekV2Lite_ShareGPT_R4_C${C}_B16_L512_O64 --timeout 5400 > $R/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
echo ALLDONE
