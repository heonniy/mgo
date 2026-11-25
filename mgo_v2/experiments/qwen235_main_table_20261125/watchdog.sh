#!/bin/bash
# Hang watchdog for run_all.sh. If the running job's progress files (phase/repeat/progress) do not change
# for longer than its threshold, send SIGABRT to its worker processes: PYTHONFAULTHANDLER=1 then writes every
# thread's Python stack into the job's run.log, and the job fails instead of hanging until its timeout.
# run_all.sh skips PASS jobs, so rerunning it retries only the aborted ones.
R=/home/hwlee/mgo-results/qwen235_main_table_20261125; LOG=$R/orchestrator.log
say() { echo "$(date +%F' '%T) $*" >> $LOG; }
threshold() { case "$1" in
  *OURS*) echo 900;;                 # OURS: load ~3 min, one repeat ~2 min
  infinity_smoke*) echo 18000;;      # first MoE() call builds the 470 GB offload store
  *) echo 9000;; esac; }             # baseline phases: up to ~35-60 min per repeat expected
desc() { local p=$1; for c in $(pgrep -P $p); do echo $c; desc $c; done; }
handled=""
while true; do
  cur=$(grep -E " STEP [^ ]+ (START|DONE)" $LOG | tail -1)
  if echo "$cur" | grep -q " START$"; then
    L=$(echo "$cur" | awk '{print $4}'); J=$R/jobs/$L
    newest=$(stat -c %Y $J/status.json $J/phase.json $J/repeat*.json $J/progress_repeat*.json 2>/dev/null | sort -n | tail -1)
    if [ -n "$newest" ]; then
      age=$(( $(date +%s) - newest )); th=$(threshold $L)
      if [ $age -gt $th ] && ! echo " $handled " | grep -q " $L "; then
        handled="$handled $L"
        top=$(python3 -c "import json;print(json.load(open('$J/status.json')).get('pid',''))" 2>/dev/null)
        pids=$( [ -n "$top" ] && desc $top | while read q; do [ "$(cat /proc/$q/comm 2>/dev/null)" = python ] || [ "$(cat /proc/$q/comm 2>/dev/null)" = python3 ] || [ "$(cat /proc/$q/comm 2>/dev/null)" = pt_main_thread ] && echo $q; done | tr '\n' ' ')
        say "WATCHDOG STALL $L no progress for ${age}s (threshold ${th}s); SIGABRT to workers [$pids] for stack dump"
        for q in $pids; do kill -ABRT $q 2>/dev/null; done
        sleep 90
        [ -n "$top" ] && kill -0 $top 2>/dev/null && { say "WATCHDOG $L still alive after SIGABRT; SIGKILL process group $top"; kill -KILL -$top 2>/dev/null; }
      fi
    fi
  fi
  grep -q "ALLDONE" $LOG && [ "$(tail -1 $LOG | grep -c ALLDONE)" = 1 ] && exit 0
  sleep 30
done
