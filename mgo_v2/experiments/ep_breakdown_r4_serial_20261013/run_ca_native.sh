#!/bin/bash
# CA_NATIVE (code 8): same objective as CA, solved by the C++ min-cost-flow kernel (scripts/native_ca_assignment.cpp); Hungarian maximization of local token demand within the balanced quota, no compute balance),
# same flags as the BR/NEAR diagnostics. A watcher logs every GPU compute process during the runs.
R=/home/hwlee/mgo-results/ep_breakdown_r4_serial_20261013
while pgrep -f "python run_headline_job.py" >/dev/null; do sleep 10; done
( while true; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ') | foreign: $(ps -eo pid,cmd | grep -E '[r]un_cell.py' | awk '{print $1}' | tr '\n' ' ')"; sleep 10; done ) > $R/ca_native_gpu_watch.log 2>&1 &
W=$!
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$R
for rep in 1 2; do
 L=c30_CA_NATIVE_d${rep}
 [ -f $R/$L/result.json ] && continue
 echo "$(date +%T) start $L"
 timeout 2400 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C30_B16_L512_O64 --repeats 1 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy CA_NATIVE --post-generation-diagnostic > $R/$L.driver.log 2>&1
 echo "$(date +%T) done $L rc=$?"
done
kill $W
echo CANDONE
