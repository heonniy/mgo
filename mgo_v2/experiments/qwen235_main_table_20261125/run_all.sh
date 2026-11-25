#!/bin/bash
# Qwen3-235B main-table cell: one chained pipeline, no manual hand-offs.
#   gguf wait (PID) -> gguf receipt -> OURS timed (main_OURS, random overlap, random serial)
#   -> MoE-Infinity smoke (builds offload store) -> DeepSpeed smoke -> llama smoke
#   -> baseline timed (Infinity, DeepSpeed, llama) -> stability extension (+2 trials if TPOT spread > 5%).
# GPU jobs run one at a time under $R/gpu.lock. Every transition is a line in $R/orchestrator.log.
GGUF_PID=$1
R=/home/hwlee/mgo-results/qwen235_main_table_20261125
PKG=/home/hwlee/mgo-qwen235/mgo_v2
T=/home/hwlee/mgo-tools/headline-r4
PY=/home/hwlee/sub-moe/phase01/.venv/bin/python
W=$R/WORKLOADS.json; CELL=Qwen3-235B_ShareGPT_R4_C30_B16_L512_O64
LOG=$R/orchestrator.log
G=$T/Qwen3-235B-A22B-Instruct-2507-BF16.gguf
say() { echo "$(date +%F' '%T) $*" >> $LOG; }
mkdir -p $R/jobs

say "STEP gguf WAIT pid=$GGUF_PID"
# PID 0 or empty means nothing to wait for (kill -0 0 would target our own process group).
[ "${GGUF_PID:-0}" -gt 0 ] && while kill -0 $GGUF_PID 2>/dev/null; do sleep 15; done
if grep -q "Model successfully exported" $T/gguf-convert-qwen235.log && [ -s $G ]; then
  say "STEP gguf DONE bytes=$(stat -c %s $G)"
  [ -s $PKG/experiments/qwen235_main_table_20261125/QWEN235_GGUF_RECEIPT.json ] || {
  say "STEP gguf_receipt START"
  python3 - <<P
import json,hashlib,os
g='$G';h=hashlib.sha256()
with open(g,'rb') as f:
    for b in iter(lambda:f.read(1<<26),b''):h.update(b)
json.dump(dict(source='/home/hwlee/model/Qwen3-235B-A22B-Instruct-2507',command=['$T/base-env/bin/python','$T/llama.cpp/convert_hf_to_gguf.py','/home/hwlee/model/Qwen3-235B-A22B-Instruct-2507','--outfile',g,'--outtype','bf16'],llama_commit='3109914090564b4c5280f30896369d55b86bbdbf',output=g,bytes=os.path.getsize(g),sha256=h.hexdigest()),open('$PKG/experiments/qwen235_main_table_20261125/QWEN235_GGUF_RECEIPT.json','w'),indent=2)
P
  say "STEP gguf_receipt DONE rc=$?"; }
else
  say "STEP gguf FAIL (see $T/gguf-convert-qwen235.log)"
fi

# job <label> <timed:0|1> <env assignments...> -- <run_headline_job args...>
job() {
  L=$1; shift
  if [ -f $R/jobs/$L/status.json ] && grep -q '"status": "PASS"' $R/jobs/$L/status.json; then say "STEP $L SKIP already PASS"; return 0; fi
  [ -d $R/jobs/$L ] && mv $R/jobs/$L $R/jobs/failed_${L}.$(date +%s)
  ENVS=(); while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
  say "STEP $L WAIT_LOCK"
  exec 9>$R/gpu.lock; flock -x 9
  say "STEP $L START"
  (cd $PKG/scripts && env MGO_HEADLINE_ROOT=$R/jobs "${ENVS[@]}" $PY -u run_headline_job.py --workloads $W --cell $CELL --label $L "$@" > $R/jobs/$L.driver.log 2>&1)
  rc=$?; flock -u 9; exec 9>&-
  st=$(python3 -c "import json;print(json.load(open('$R/jobs/$L/status.json'))['status'])" 2>/dev/null)
  say "STEP $L DONE rc=$rc status=$st $(summ $L)"
  [ "$st" = PASS ]
}
summ() { python3 - "$R/jobs/$1" <<'P' 2>/dev/null
import json,sys,glob,os
d=sys.argv[1];rows=[]
for k in range(1,10):
    f=os.path.join(d,f'repeat{k}.json')
    if os.path.exists(f):rows.append(json.load(open(f)))
if rows:print('TPOT='+','.join('%.3f'%r['TPOT'] for r in rows),'TTFT='+','.join('%.2f'%r['TTFT'] for r in rows))
P
}
spread_gt5() { python3 - "$@" <<'P'
import json,sys,os
t=[]
for d in sys.argv[1:]:
    for k in range(1,10):
        f=os.path.join(d,f'repeat{k}.json')
        if os.path.exists(f):t.append(json.load(open(f))['TPOT'])
sys.exit(0 if t and max(t)/min(t)-1>0.05 else 1)
P
}
timed() { # label system-args... ; runs 3 trials, adds 2 more if TPOT spread > 5%
  L=$1; shift
  job $L "$@" || return 1
  local args=("$@"); args=("${args[@]/--repeats 3/}")
  if spread_gt5 $R/jobs/$L; then
    say "STEP $L UNSTABLE (TPOT spread > 5%): adding 2 trials"
    local i; local new=(); local skip=0
    for x in "$@"; do if [ $skip = 1 ]; then new+=(2); skip=0; elif [ "$x" = "--repeats" ]; then new+=("$x"); skip=1; else new+=("$x"); fi; done
    job ${L}_ext2 "${new[@]}"
  fi
}

OURS="--expert-executor native --native-prefill --prefetch-off --prefill-optimized --prefill-layout-fast --decode-layout-fast --expert-source shared_pinned"
timed q235_main_OURS_r3 -- --system main_OURS --worker headline_ours_worker.py --ours-final --expert-source shared_pinned --repeats 3 --timeout 7200
timed q235_OURS_random_overlap_r3 MGO_RANDOM_SALT=1 -- --system OURS_random_overlap --worker headline_ours_worker.py $OURS --policy RANDOM --decode-policy RANDOM --repeats 3 --timeout 7200
timed q235_OURS_random_serial_r3 MGO_RANDOM_SALT=1 -- --system OURS_random_serial --worker headline_ours_worker.py $OURS --policy RANDOM --decode-policy RANDOM --h2d-serial-ablation --repeats 3 --timeout 7200

INF="--system MoE-Infinity-repaired --worker headline_infinity_worker.py --python $T/infinity-env/bin/python --ranks 1"
DS="--system DeepSpeed-ZeRO-Inference --worker headline_deepspeed_worker.py --python $T/base-env/bin/python"
LL="--system llama.cpp-sync-balanced7 --worker headline_llama_qwen235_sync_worker.py --python $T/base-env/bin/python --ranks 1 --llama-threads 32 --llama-cuda-graphs off --llama-graph-reuse off --llama-expert-placement balanced7"
job infinity_smoke_v1 -- $INF --smoke --timeout 21600 && inf_ok=1
job deepspeed_smoke_v1 -- $DS --smoke --timeout 14400 && ds_ok=1
[ -s $G ] && job llama_smoke_v1 -- $LL --smoke --timeout 28800 && ll_ok=1
[ "$inf_ok" = 1 ] && timed q235_infinity_r3 -- $INF --repeats 3 --timeout 28800
[ "$ds_ok" = 1 ] && timed q235_deepspeed_r3 -- $DS --repeats 3 --timeout 36000
[ "$ll_ok" = 1 ] && timed q235_llama_r3 -- $LL --repeats 3 --timeout 36000
say "ALLDONE"
