#!/bin/bash
# Worst-salt BR_SALT (code 26) from the trace-based CPU search: B32 salt 183, B64 salt 57; same diagnostic flags.
R=/home/hwlee/mgo-results/ep_breakdown_batch_20261013
while pgrep -f "python run_headline_job.py" >/dev/null; do sleep 10; done
cd /home/hwlee/mgo-haq/mgo_v2/scripts
export MGO_HEADLINE_ROOT=$R
for spec in 32:183 64:57; do
 B=${spec%%:*}; S=${spec#*:}
 for rep in 1 2; do
  L=b${B}_BR_SALT${S}_d${rep}
  [ -f $R/$L/result.json ] && continue
  echo "$(date +%T) start $L"
  MGO_RANDOM_SALT=$S timeout 3600 /home/hwlee/sub-moe/phase01/.venv/bin/python run_headline_job.py --workloads $R/WORKLOADS.json --label $L --system ours --worker headline_ours_worker.py --cell Qwen3_ShareGPT_R4_C30_B${B}_L512_O64 --repeats 1 --expert-executor native --native-prefill --prefetch-off --h2d-serial-ablation --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy BR --decode-policy BR_SALT --post-generation-diagnostic --timeout 3600 > $R/$L.driver.log 2>&1
  echo "$(date +%T) done $L rc=$?"
 done
done
echo SALTDONE
