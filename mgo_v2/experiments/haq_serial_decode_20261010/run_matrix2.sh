#!/bin/bash
# v2 driver: BR / LA_CA_NEAR / HAQ / BW, forward then reverse order. Skips finished labels.
R=/home/hwlee/mgo-results/haq_serial_decode_20261010
while pgrep -f "run_headline_job.py --workloads" >/dev/null; do sleep 10; done
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$R
for C in 50 30; do
 for spec in "1:BR LA_CA_NEAR HAQ BW" "2:BW HAQ LA_CA_NEAR BR"; do
  round=${spec%%:*}; ORDER=${spec#*:}
  for POL in $ORDER; do
   L=c${C}_${POL}_r${round}
   [ -f $R/$L/result.json ] && continue
   [ -d $R/$L ] && mv $R/$L $R/${L}.failed.$(date +%s)
   echo "$(date +%T) start $L"
   timeout 2400 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C${C}_B16_L512_O64 --repeats 3 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy $POL > $R/$L.driver.log 2>&1
   echo "$(date +%T) done $L rc=$?"
  done
 done
done
echo ALLDONE
