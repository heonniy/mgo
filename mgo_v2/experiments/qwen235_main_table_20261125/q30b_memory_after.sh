#!/bin/bash
# After the MoE-Infinity job (wait on its orchestrator PID), measure Qwen3-30B main_OURS on the main-table cell
# with the memory-footprint receipt. 30B keeps the main-table rank-private full pinned source (default).
R=/home/hwlee/mgo-results/qwen235_main_table_20261125; PKG=/home/hwlee/mgo-qwen235/mgo_v2; LOG=$R/orchestrator.log
W=/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json; L=q30b_main_OURS_memory_r3
while kill -0 $1 2>/dev/null; do sleep 15; done
say() { echo "$(date +%F' '%T) $*" >> $LOG; }
say "STEP $L WAIT_LOCK"; exec 9>$R/gpu.lock; flock -x 9; say "STEP $L START"
(cd $PKG/scripts && env MGO_HEADLINE_ROOT=$R/jobs /home/hwlee/sub-moe/phase01/.venv/bin/python -u run_headline_job.py --workloads $W --cell Qwen3_ShareGPT_R4_C30_B16_L512_O64 --label $L --system main_OURS --worker headline_ours_worker.py --ours-final --repeats 3 --timeout 7200 > $R/jobs/$L.driver.log 2>&1)
rc=$?; flock -u 9
st=$(python3 -c "import json;print(json.load(open('$R/jobs/$L/status.json'))['status'])" 2>/dev/null)
tp=$(python3 -c "import json,os;print(','.join('%.3f'%json.load(open(f'$R/jobs/$L/repeat{k}.json'))['TPOT'] for k in range(1,4) if os.path.exists(f'$R/jobs/$L/repeat{k}.json')))" 2>/dev/null)
say "STEP $L DONE rc=$rc status=$st TPOT=$tp"; say "ALLDONE"
