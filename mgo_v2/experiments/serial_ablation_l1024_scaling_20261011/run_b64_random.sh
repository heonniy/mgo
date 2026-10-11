#!/bin/bash
# OURS-random (RANDOM, code 19) only, local B64, R4 C30, same serial ablation flags as run_lmsys.sh.
# 8 cells: ShareGPT/LMSYS-Chat-1M x Qwen3/DeepSeek x input 512/1024 (main-table manifests).
# Two rounds per cell (MGO_RANDOM_SALT = round). Input 1024 first so memory problems surface early.
R=/home/hwlee/mgo-results/serial_ablation_l1024_scaling_20261011
PY=/home/hwlee/sub-moe/phase01/.venv/bin/python
MT=/home/hwlee/mgo-results/main_table_2x2_20261008
PKG=/home/hwlee/mgo-haq/mgo_v2
export MGO_HEADLINE_ROOT=$R
SERIAL="--expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR"

headline() { # label, then the command
 L=$1; shift
 [ -f $R/$L/result.json ] && return
 [ -d $R/$L ] && mv $R/$L $R/failed_${L}.$(date +%s)
 echo "$(date +%T) start $L"; "$@" > $R/$L.driver.log 2>&1; rc=$?; echo "$(date +%T) done $L rc=$rc"
}
stop() { [ -f $R/STOP ] && { echo STOPPED; exit 1; }; }

ds() { # dataset-dir tag input round
 stop; cd $PKG
 MGO_RANDOM_SALT=$4 MGO_DS_DECODE_POLICY_SCHEDULE=RANDOM,RANDOM MGO_DS_H2D_SERIAL_ABLATION=1 MGO_DS_PREFILL_POLICY=BR \
 PATH=/home/hwlee/mgo-tools/native-expert-build/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1 NUMBA_CACHE_DIR=/tmp/mgo-haq-deepseek-numba-v1 \
 headline ds_$2l$3_b64_r4_RANDOM_r$4 timeout 5600 $PY scripts/run_headline_job.py --workloads $MT/$1/WORKLOADS.json --label ds_$2l$3_b64_r4_RANDOM_r$4 --system main_OURS --worker headline_ours_deepseek_multipolicy_worker.py --cell DeepSeekV2Lite_$1_R4_C30_B64_L$3_O64 --timeout 5400
}
qwen() { # dataset-dir tag input round
 stop; cd $PKG/scripts
 MGO_RANDOM_SALT=$4 headline qwen_$2l$3_b64_r4_RANDOM_r$4 timeout 3600 $PY run_headline_job.py --workloads $MT/$1/WORKLOADS.json --label qwen_$2l$3_b64_r4_RANDOM_r$4 --system ours --worker headline_ours_worker.py --cell Qwen3_$1_R4_C30_B64_L$3_O64 --repeats 3 $SERIAL --decode-policy RANDOM
}

for n in 1024 512; do
 for r in 1 2; do
  ds ShareGPT "" $n $r; qwen ShareGPT "" $n $r
  ds LMSYS-Chat-1M lmsys_ $n $r; qwen LMSYS-Chat-1M lmsys_ $n $r
 done
done
echo ALLDONE
