#!/bin/bash
# Serial live TPOT runs (main-table Ours config, --ours-final, 3 repeats); only Step-4 victim score changes.
set -uo pipefail
cd /home/hwlee/mgo-ferry-appendix/mgo_v2
export MGO_HEADLINE_ROOT=/home/hwlee/mgo-results/ferry_gate_score_20261013/tpot
mkdir -p $MGO_HEADLINE_ROOT
until grep -q "=== DONE" /home/hwlee/mgo-results/ferry_gate_score_20261013/captures.log 2>/dev/null; do sleep 30; done
run() { # score window batch pct
  label=ftpot_${1}_W${2}_B${3}_C${4}_v1
  [ -f $MGO_HEADLINE_ROOT/$label/status.json ] && grep -q '"PASS"' $MGO_HEADLINE_ROOT/$label/status.json && return
  echo "=== $label $(date)"
  MGO_FERRY_EVICT_SCORE=$1 MGO_FERRY_GATE_WINDOW=$2 /home/hwlee/sub-moe/phase01/.venv/bin/python -u scripts/run_headline_job.py \
    --workloads experiments/ferry_gate_score_20261013/workloads/TPOT_WORKLOADS.json --label $label \
    --cell Qwen3_ShareGPT_R4_C${4}_B${3}_L512_O64 --repeats 3 --system Ours --worker headline_ours_worker.py --ours-final --timeout 7200 \
    || echo "FAIL $label"
  echo "{\"evict_score\":\"$1\",\"gate_window\":$2,\"local_batch\":$3,\"cache_percent\":$4}" > $MGO_HEADLINE_ROOT/$label/ferry_policy.json
}
for pct in 30 60; do for b in 16 64 8; do for s in full count selected lru; do run $s 128 $b $pct; done; done; done
for b in 16 64; do for w in 32 512; do run full $w $b 30; done; done
echo "=== DONE $(date)"
