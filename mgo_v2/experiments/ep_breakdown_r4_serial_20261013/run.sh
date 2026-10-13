#!/bin/bash
# R4 serial ablation, BR vs LA_CA_NEAR, with post-generation per-layer diagnostic (EP cost breakdown).
R=/home/hwlee/mgo-results/ep_breakdown_r4_serial_20261013
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$R
for C in 30; do
 for spec in BR:1 LA_CA_NEAR:1 LA_CA_NEAR:2 BR:2; do
  POL=${spec%%:*}; rep=${spec#*:}; L=c${C}_${POL}_d${rep}
  [ -f $R/$L/result.json ] && continue
  echo "$(date +%T) start $L"
  timeout 2400 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C${C}_B16_L512_O64 --repeats 1 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy $POL --post-generation-diagnostic > $R/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
echo EPDONE
