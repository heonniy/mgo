#!/bin/bash
# Wait for the running pipeline (by PID), then rerun run_all.sh: PASS jobs are skipped, so only failed or
# skipped systems run again. Restarts the watchdog for the new pipeline. PIDs are taken from $!.
R=/home/hwlee/mgo-results/qwen235_main_table_20261125; E=/home/hwlee/mgo-qwen235/mgo_v2/experiments/qwen235_main_table_20261125
WAIT_PID=$1
while kill -0 $WAIT_PID 2>/dev/null; do sleep 15; done
echo "$(date +%F' '%T) --- rerun run_all.sh (retry failed/skipped systems; PASS jobs skipped)" >> $R/orchestrator.log
$E/run_all.sh 0 > $R/orchestrator.stdout 2>&1 < /dev/null &
echo $! > $R/orchestrator.pid
sleep 5
kill -0 $(cat $R/watchdog.pid) 2>/dev/null || { $E/watchdog.sh > $R/watchdog.stdout 2>&1 < /dev/null & echo $! > $R/watchdog.pid; }
wait
