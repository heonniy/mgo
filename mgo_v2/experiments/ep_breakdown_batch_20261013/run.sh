#!/bin/bash
# R4 C30 serial ablation, BR vs LA_CA_NEAR at local B32 and B64, with post-generation per-layer diagnostic.
R=/home/hwlee/mgo-results/ep_breakdown_batch_20261013
while pgrep -f "python run_headline_job.py" >/dev/null; do sleep 10; done
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$R
for B in 32 64; do
 for spec in BR:1 LA_CA_NEAR:1 LA_CA_NEAR:2 BR:2; do
  POL=${spec%%:*}; rep=${spec#*:}; L=b${B}_${POL}_d${rep}
  [ -f $R/$L/result.json ] && continue
  echo "$(date +%T) start $L"
  timeout 3600 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads $R/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C30_B${B}_L512_O64 --repeats 1 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy $POL --post-generation-diagnostic --timeout 3600 > $R/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
echo BATCHDONE
