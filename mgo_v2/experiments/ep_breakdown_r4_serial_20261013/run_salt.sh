#!/bin/bash
# Worst-salt BR (BR_SALT, code 26, salt 46 from CPU search) at R4 C30, same diagnostic flags; after run.sh.
R=/home/hwlee/mgo-results/ep_breakdown_r4_serial_20261013
until grep -q EPDONE $R/run_c30.log 2>/dev/null; do sleep 20; done
while pgrep -f "python run_headline_job.py" >/dev/null; do sleep 10; done
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$R
for rep in 1 2; do
 L=c30_BR_SALT46_d${rep}
 [ -f $R/$L/result.json ] && continue
 echo "$(date +%T) start $L"
 MGO_RANDOM_SALT=46 timeout 2400 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads /home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C30_B16_L512_O64 --repeats 1 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy BR_SALT --post-generation-diagnostic > $R/$L.driver.log 2>&1
 echo "$(date +%T) done $L rc=$?"
done
echo SALTDONE
