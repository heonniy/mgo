#!/bin/bash
# End-to-end validation suite for the cooperative-offloading EP path
# (4-rank torchrun, real Qwen3-30B).  Validates the 2026-05-28 changes:
# slot pool + per-expert chunk pipeline + chunk1_staging + fast-a2a.
#
# Each scenario cold-starts (full cleanup), runs via the retry watchdog, and
# asserts on run.log + trace JSON.
#
#   T1  cap=inf naive   staging off,on  -> correctness, evict=0, drift=0
#   T2  cap=32  naive   staging off      -> LRU eviction, drift=0
#   T4  cap=8   balanced staging on      -> low-cap balanced generate, drift=0
#   T6  cap=1   naive   staging on       -> staging promotion / view sync
#
# (T3 hit-victim is a pure-Python controller unit test: tests/test_coop_e2e.py
#  ::test_hit_victim_drift_free.  T5 batch<world / 0-token NCCL symmetry is
#  covered by tests/test_fast_a2a.py empty/zero-rank cases.)
#
# Usage:  bash tests/run_e2e_validation.sh [T1 T2 T4 T6]   (default: all)
set -u
EP_DIR=/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP
SWEEP="$EP_DIR/configs/_sweep"
RESULTS=/home/work/hyewon.lee/실험/main_exp/results
export MAX_RETRIES="${MAX_RETRIES:-2}"
MAXNEW="${MAXNEW:-8}"

cold_cleanup() {
    pkill -9 -f 'torchrun|m4_long_gen|numa_wrap' 2>/dev/null || true
    sleep 5
    rm -f /tmp/moe_numa*.sock /tmp/*.init.lock 2>/dev/null
    rm -f /home/work/hyewon.lee/cache/moe_offload_qwen3_30b*/.init.lock 2>/dev/null
    local t=0
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 500 ] || [ $t -ge 40 ]; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
        sleep 3; t=$((t+3))
    done
    sync; sleep 2
}

# run_case <name> <config> <staging 0|1> <extra_env>
run_case() {
    local name="$1" cfg="$2" staging="$3" extra="${4:-}"
    cold_cleanup
    local tag="E2E_${name}"
    local logf="/tmp/e2e_${name}.log"
    echo ""
    echo "######## $name : $(basename "$cfg") staging=$staging $extra ########"
    ( export MOE_EP_CHUNK1_STAGING="$staging"; export MOE_EP_HEAVY_DEBUG=1; eval "$extra"
      bash /tmp/watchdog_run.sh "$cfg" "$tag" "$MAXNEW" ) > "$logf" 2>&1
    # locate the run dir
    local d
    d=$(ls -dt --color=never "$RESULTS/${tag}"_try*_* 2>/dev/null | head -1)
    [ -z "$d" ] && { echo "[$name] FAIL: no run dir"; return 1; }
    local rl="$d/run.log"
    # --- hard assertions ---
    local drift gen fatal verdict="PASS"
    drift=$(grep -cE "DRIFT" "$rl" 2>/dev/null || echo 99)
    gen=$(grep -cE "^\[rank[0-9]\] generate " "$rl" 2>/dev/null || echo 0)
    fatal=$(grep -cE "batch_size should be|SIGABRT|Signal 6|Traceback|All cached expert locked|view Byte" "$rl" 2>/dev/null || echo 0)
    [ "$drift" -ne 0 ] && verdict="FAIL(drift=$drift)"
    [ "$gen" -lt 1 ] && verdict="FAIL(no-generate)"
    [ "$fatal" -ne 0 ] && verdict="FAIL(fatal=$fatal)"
    # --- soft reports ---
    local stg_eng evict
    stg_eng=$(grep -c "STAGING PATH engaged" "$rl" 2>/dev/null || echo 0)
    local jf
    jf=$(ls -t --color=never /tmp/moe_ep_traces/${tag}_*.json 2>/dev/null | head -1)
    evict="n/a"
    [ -n "$jf" ] && evict=$(python3 -c "import json;print(json.load(open('$jf'))['totals'].get('cache_evictions','n/a'))" 2>/dev/null || echo err)
    echo "[$name] verdict=$verdict  drift=$drift gen=$gen fatal=$fatal | soft: evict=$evict staging_engaged=$stg_eng"
    echo "[$name] logdir=$d"
    # scenario-specific soft checks
    case "$name" in
      T1*) [ "$evict" != "0" ] && [ "$evict" != "n/a" ] && echo "[$name] WARN: cap=inf expected evict=0, got $evict";;
      T6*) [ "$stg_eng" -lt 1 ] && echo "[$name] WARN: staging path did not engage (expected overflow at cap=1)";;
    esac
    [ "$verdict" = "PASS" ] && return 0 || return 1
}

SEL="${*:-R2 T1off T1on T2 T4 T6}"
declare -A RES
for t in $SEL; do
  case "$t" in
    R2)    run_case "R2_cap8_naive"      "$SWEEP/val_R2_cap8_naive.yaml"      1; RES[$t]=$?;;
    T1off) run_case "T1off_capinf_naive" "$SWEEP/val_R1_unlimited_naive.yaml" 0; RES[$t]=$?;;
    T1on)  run_case "T1on_capinf_naive"  "$SWEEP/val_R1_unlimited_naive.yaml" 1; RES[$t]=$?;;
    T2)    run_case "T2_cap32_naive"     "$SWEEP/val_R2_cap32_naive.yaml"     0; RES[$t]=$?;;
    T4)    run_case "T4_cap8_balanced"   "$SWEEP/val_R3_cap8_balanced.yaml"   1; RES[$t]=$?;;
    T6)    run_case "T6_cap1_naive"      "$SWEEP/val_R2_cap1_naive.yaml"      1; RES[$t]=$?;;
    *) echo "unknown test $t";;
  esac
done

echo ""
echo "================ E2E VALIDATION SUMMARY ================"
fail=0
for t in $SEL; do
  if [ "${RES[$t]:-1}" -eq 0 ]; then echo "  $t: PASS"; else echo "  $t: FAIL"; fail=1; fi
done
cold_cleanup
exit $fail
