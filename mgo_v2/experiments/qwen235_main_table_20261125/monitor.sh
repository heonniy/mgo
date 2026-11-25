#!/bin/bash
# Event stream for the Qwen3-235B pipeline: orchestrator transitions, post-job HEALTH line,
# STALL warning (no GPU use, progress-file update or Infinity store growth for 10 min), 15-min heartbeat.
R=/home/hwlee/mgo-results/qwen235_main_table_20261125; OP=$(cat $R/orchestrator.pid)
HC=/home/hwlee/mgo-qwen235/mgo_v2/experiments/qwen235_main_table_20261125/health_check.sh
ST=/home/hwlee/mgo-tools/headline-r4/infinity-qwen235-bf16-store
tail -n 0 -F $R/orchestrator.log | while read -r line; do
  echo "$line"
  case "$line" in *" DONE rc="*) L=$(echo "$line" | awk '{print $4}'); sleep 5; $HC "$L";; esac
done &
last=$(date +%s); busy=$(date +%s); stall_sent=0; prev_store=0
while true; do
  now=$(date +%s)
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  util=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '($1==0||$1==1||$1==4||$1==5)&&$2>0{n++}END{print n+0}')
  lg=$(ls -t $R/jobs/*/run.log 2>/dev/null | head -1); lgage=$([ -n "$lg" ] && echo $((now-$(stat -c %Y "$lg"))) || echo 99999)
  ph=$(ls -t $R/jobs/*/phase.json $R/jobs/*/progress_repeat*.json 2>/dev/null | head -1); phage=$([ -n "$ph" ] && echo $((now-$(stat -c %Y "$ph"))) || echo 99999)
  store=$(du -sb $ST 2>/dev/null | cut -f1); store=${store:-0}
  if [ "$procs" -eq 0 ] || [ "$util" -gt 0 ] || [ "$phage" -lt 600 ] || [ "$lgage" -lt 600 ] || [ "$store" -gt "$prev_store" ]; then busy=$now; stall_sent=0; fi
  prev_store=$store
  if [ $((now-busy)) -ge 600 ] && [ $stall_sent = 0 ]; then
    echo "STALL? $(date +%T) gpu procs=$procs, GPUs 0%, no progress file, run.log or store growth for $(( (now-busy)/60 )) min; last=$(tail -1 $R/orchestrator.log | cut -c12-)"
    stall_sent=1
  fi
  if [ $((now-last)) -ge 900 ]; then
    gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$1==0||$1==1||$1==4||$1==5{printf "%s:%sMiB/%s%% ",$1,$2,$3}')
    phs=$([ -n "$ph" ] && python3 -c "import json;d=json.load(open('$ph'));print('$ph'.split('/')[-2],d.get('phase'),d.get('repeat',''),d.get('step',''),d.get('tokens',''))" 2>/dev/null)
    echo "HEARTBEAT $(date +%T) orchestrator=$(kill -0 $OP 2>/dev/null && echo alive || echo EXITED) job=[$phs] infinity_store=$((store/1000000000))GB gpu[$gpu] avail=$(awk '$1=="MemAvailable:"{printf "%.0f",$2/1048576}' /proc/meminfo)GiB last=$(tail -1 $R/orchestrator.log | cut -c12-)"
    last=$now
  fi
  kill -0 $OP 2>/dev/null || { sleep 8; echo "ORCHESTRATOR EXITED: $(tail -1 $R/orchestrator.log)"; exit 0; }
  sleep 20
done
