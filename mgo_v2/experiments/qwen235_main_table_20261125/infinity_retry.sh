#!/bin/bash
# MoE-Infinity timed cell with retries. The repaired MoE-Infinity build has an intermittent native segfault
# (exit -11) in warmup prefill, also seen and cleared by a plain retry in qwen_cache_ablation_20261009
# (BASELINE_FAILURES.md). Same configuration on every attempt; failed attempts are kept (failed_* dirs) and
# excluded. Adds 2 trials if the 3 TPOTs spread by more than 5%.
R=/home/hwlee/mgo-results/qwen235_main_table_20261125; PKG=/home/hwlee/mgo-qwen235/mgo_v2; T=/home/hwlee/mgo-tools/headline-r4
PY=/home/hwlee/sub-moe/phase01/.venv/bin/python; W=$R/WORKLOADS.json; CELL=Qwen3-235B_ShareGPT_R4_C30_B16_L512_O64; LOG=$R/orchestrator.log
INF="--system MoE-Infinity-repaired --worker headline_infinity_worker.py --python $T/infinity-env/bin/python --ranks 1"
say() { echo "$(date +%F' '%T) $*" >> $LOG; }
run() { # label repeats
  L=$1
  [ -d $R/jobs/$L ] && mv $R/jobs/$L $R/jobs/failed_${L}.$(date +%s)
  say "STEP $L WAIT_LOCK"; exec 9>$R/gpu.lock; flock -x 9; say "STEP $L START"
  (cd $PKG/scripts && env MGO_HEADLINE_ROOT=$R/jobs $PY -u run_headline_job.py --workloads $W --cell $CELL --label $L $INF --repeats $2 --timeout 28800 > $R/jobs/$L.driver.log 2>&1)
  rc=$?; flock -u 9; exec 9>&-
  st=$(python3 -c "import json;print(json.load(open('$R/jobs/$L/status.json'))['status'])" 2>/dev/null)
  tp=$(python3 -c "import json,os;print(','.join('%.3f'%json.load(open(f'$R/jobs/$L/repeat{k}.json'))['TPOT'] for k in range(1,6) if os.path.exists(f'$R/jobs/$L/repeat{k}.json')))" 2>/dev/null)
  say "STEP $L DONE rc=$rc status=$st TPOT=$tp"; [ "$st" = PASS ]
}
ok=0; for a in 1 2 3; do run q235_infinity_r3 3 && { ok=1; break; }; say "STEP q235_infinity_r3 RETRY attempt $a failed"; done
if [ $ok = 1 ] && python3 - <<'P'
import json,sys
t=[json.load(open(f'/home/hwlee/mgo-results/qwen235_main_table_20261125/jobs/q235_infinity_r3/repeat{k}.json'))['TPOT'] for k in (1,2,3)]
sys.exit(0 if max(t)/min(t)-1>0.05 else 1)
P
then say "STEP q235_infinity_r3 UNSTABLE (TPOT spread > 5%): adding 2 trials"; for a in 1 2 3; do run q235_infinity_r3_ext2 2 && break; done; fi
say "ALLDONE"
