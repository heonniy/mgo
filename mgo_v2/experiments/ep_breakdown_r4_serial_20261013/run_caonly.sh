#!/bin/bash
# CA only = LA_CA_NEAR code path with the balance filter disabled (MGO_NEAR_BPS=1e9: every rank with quota
# left is a candidate, so placement is pure argmax_r d_{r,e} locality). Separate numba cache so the frozen
# NEAR_BPS global is recompiled. Same flags as the BR/NEAR diagnostics. A watcher logs foreign GPU users.
R=/home/hwlee/mgo-results/ep_breakdown_r4_serial_20261013
while pgrep -f "python run_headline_job.py" >/dev/null; do sleep 10; done
( while true; do date +%T; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; ps -eo pid,cmd | grep -E "[r]un_cell.py" | cut -c1-80; sleep 10; done ) > $R/caonly_gpu_watch.log 2>&1 &
W=$!
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$R MGO_NEAR_BPS=1000000000 NUMBA_CACHE_DIR=/tmp/claude-1003/-home-hwlee/49004950-fa4e-4ec3-9037-4689892fd6ff/scratchpad/caonly/nb_runtime
for rep in 1 2; do
 L=c30_CA_ONLY_d${rep}
 [ -f $R/$L/result.json ] && continue
 echo "$(date +%T) start $L"
 timeout 2400 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C30_B16_L512_O64 --repeats 1 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy LA_CA_NEAR --post-generation-diagnostic > $R/$L.driver.log 2>&1
 echo "$(date +%T) done $L rc=$?"
done
kill $W
echo CADONE
