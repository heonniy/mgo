#!/bin/bash
# Serial ablation (same flags as ../cache_size_sweep_ablation/run_*.sh): OURS (HAQ) vs RANDOM decode, BR prefill,
# native serial executor + --h2d-serial-ablation, prefetch off, C30, ShareGPT, local B16, O64.
# Part A: R4 input 1024 (main-table manifest), Qwen3 and DeepSeek-V2-Lite.
# Part B: Qwen3 input 512 at R2 (GPUs 0,1), R4 (0,1,4,5), R8 (0-7).
# Each cell runs HAQ r1, RANDOM r1, RANDOM r2, HAQ r2 (ABBA); MGO_RANDOM_SALT = round. Skips finished jobs.
R=/home/hwlee/mgo-results/serial_ablation_l1024_scaling_20261011
PY=/home/hwlee/sub-moe/phase01/.venv/bin/python
MAIN=/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json
QCACHE=/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json
PKG=/home/hwlee/mgo-haq/mgo_v2
ORDER="HAQ:1 RANDOM:1 RANDOM:2 HAQ:2"
export MGO_HEADLINE_ROOT=$R
SERIAL="--expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR"

headline() { # label, then run_headline_job args
 L=$1; shift
 [ -f $R/$L/result.json ] && return
 [ -d $R/$L ] && mv $R/$L $R/failed_${L}.$(date +%s)
 echo "$(date +%T) start $L"; "$@" > $R/$L.driver.log 2>&1; echo "$(date +%T) done $L rc=$?"
}

stop() { [ -f $R/STOP ] && { echo STOPPED; exit 1; }; }

# Part A: Qwen3 R4 L1024
cd $PKG/scripts
for spec in $ORDER; do P=${spec%%:*}; r=${spec#*:}; stop
 MGO_RANDOM_SALT=$r headline qwen_l1024_r4_${P}_r$r timeout 3600 $PY run_headline_job.py --workloads $MAIN --label qwen_l1024_r4_${P}_r$r --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C30_B16_L1024_O64 --repeats 3 $SERIAL --decode-policy $P
done
# Part A: DeepSeek R4 L1024
cd $PKG
for spec in $ORDER; do P=${spec%%:*}; r=${spec#*:}; stop
 MGO_RANDOM_SALT=$r MGO_DS_DECODE_POLICY_SCHEDULE=$P,$P MGO_DS_H2D_SERIAL_ABLATION=1 MGO_DS_PREFILL_POLICY=BR \
 PATH=/home/hwlee/mgo-tools/native-expert-build/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1 NUMBA_CACHE_DIR=/tmp/mgo-haq-deepseek-numba-v1 \
 headline ds_l1024_r4_${P}_r$r timeout 5600 $PY scripts/run_headline_job.py --workloads $MAIN --label ds_l1024_r4_${P}_r$r --system main_OURS --worker headline_ours_deepseek_multipolicy_worker.py --cell DeepSeekV2Lite_ShareGPT_R4_C30_B16_L1024_O64 --timeout 5400
done
# Part B: Qwen3 L512 R4 (cache-ablation manifest, C30 cell)
cd $PKG/scripts
for spec in $ORDER; do P=${spec%%:*}; r=${spec#*:}; stop
 MGO_RANDOM_SALT=$r headline qwen_l512_r4_${P}_r$r timeout 3600 $PY run_headline_job.py --workloads $QCACHE --label qwen_l512_r4_${P}_r$r --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C30_B16_L512_O64 --repeats 3 $SERIAL --decode-policy $P
done
# Part B: Qwen3 L512 R2 and R8 (their frozen manifests; outputs under <root>/jobs/sab_<policy>_full_v<round>)
cd $PKG/..
for G in 2 8; do
 ROOT=$( [ $G = 2 ] && echo /home/hwlee/mgo-results/qwen_r2_sharegpt_b16_l512_20261010 || echo /home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009 )
 for spec in $ORDER; do P=${spec%%:*}; r=${spec#*:}; stop
  lab=sab_$(echo $P | tr A-Z a-z); out=$ROOT/jobs/${lab}_full_v$r
  [ -f $out/result.json ] && continue
  echo "$(date +%T) start r$G $lab v$r"
  MGO_RANDOM_SALT=$r timeout 3600 $PY -u $PKG/scripts/run_qwen_r${G}_job.py --system ours --ours-mode A --ours-policy BR --decode-policy $P --h2d-serial-ablation --repeats 3 --attempt $r --job-label $lab > $R/qwen_l512_r${G}_${P}_r$r.driver.log 2>&1
  echo "$(date +%T) done r$G $lab v$r rc=$?"
 done
done
echo ALLDONE
