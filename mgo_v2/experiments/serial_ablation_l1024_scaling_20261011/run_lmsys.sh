#!/bin/bash
# Follow-up to run.sh: fill the R4 dataset x model x input matrix with the same serial ablation flags.
# Cells: DeepSeek ShareGPT L512, and LMSYS-Chat-1M L512/L1024 for Qwen3 and DeepSeek (main-table manifests).
# Qwen3 ShareGPT L512 is already in run.sh (the cache-ablation manifest has identical request hashes).
# Each cell runs HAQ r1, RANDOM r1, RANDOM r2, HAQ r2 (ABBA); MGO_RANDOM_SALT = round. Skips finished jobs.
R=/home/hwlee/mgo-results/serial_ablation_l1024_scaling_20261011
PY=/home/hwlee/sub-moe/phase01/.venv/bin/python
MT=/home/hwlee/mgo-results/main_table_2x2_20261008
PKG=/home/hwlee/mgo-haq/mgo_v2
ORDER="HAQ:1 RANDOM:1 RANDOM:2 HAQ:2"
export MGO_HEADLINE_ROOT=$R
SERIAL="--expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR"

headline() { # label, then the command
 L=$1; shift
 [ -f $R/$L/result.json ] && return
 [ -d $R/$L ] && mv $R/$L $R/failed_${L}.$(date +%s)
 echo "$(date +%T) start $L"; "$@" > $R/$L.driver.log 2>&1; echo "$(date +%T) done $L rc=$?"
}
stop() { [ -f $R/STOP ] && { echo STOPPED; exit 1; }; }

ds() { # dataset-dir tag input
 cd $PKG
 for spec in $ORDER; do P=${spec%%:*}; r=${spec#*:}; stop
  MGO_RANDOM_SALT=$r MGO_DS_DECODE_POLICY_SCHEDULE=$P,$P MGO_DS_H2D_SERIAL_ABLATION=1 MGO_DS_PREFILL_POLICY=BR \
  PATH=/home/hwlee/mgo-tools/native-expert-build/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1 NUMBA_CACHE_DIR=/tmp/mgo-haq-deepseek-numba-v1 \
  headline ds_$2l$3_r4_${P}_r$r timeout 5600 $PY scripts/run_headline_job.py --workloads $MT/$1/WORKLOADS.json --label ds_$2l$3_r4_${P}_r$r --system main_OURS --worker headline_ours_deepseek_multipolicy_worker.py --cell DeepSeekV2Lite_$1_R4_C30_B16_L$3_O64 --timeout 5400
 done
}
qwen() { # dataset-dir tag input
 cd $PKG/scripts
 for spec in $ORDER; do P=${spec%%:*}; r=${spec#*:}; stop
  MGO_RANDOM_SALT=$r headline qwen_$2l$3_r4_${P}_r$r timeout 3600 $PY run_headline_job.py --workloads $MT/$1/WORKLOADS.json --label qwen_$2l$3_r4_${P}_r$r --system ours --worker headline_ours_worker.py --cell Qwen3_$1_R4_C30_B16_L$3_O64 --repeats 3 $SERIAL --decode-policy $P
 done
}

ds ShareGPT "" 512
ds LMSYS-Chat-1M lmsys_ 512
ds LMSYS-Chat-1M lmsys_ 1024
qwen LMSYS-Chat-1M lmsys_ 512
qwen LMSYS-Chat-1M lmsys_ 1024
echo ALLDONE
