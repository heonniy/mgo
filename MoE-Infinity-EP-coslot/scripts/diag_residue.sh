#!/usr/bin/env bash
# ============================================================================
# diag_residue.sh — cross-run pinned / NUMA-shm / CUDA-registration residue
#                    diagnosis for the "repeated run → cgroup OOM" hypothesis.
#
# READ-ONLY by default. This script NEVER kills processes. It only reads cgroup
# v2 memory.* , scans /proc for leftover MoE ranks + memfd fds, and queries
# nvidia-smi. The optional `run` wrapper snapshots before/after a command and
# emits a CSV row matching the diagnosis table.
#
# Usage:
#   diag_residue.sh snapshot <label>           # one labeled snapshot
#   diag_residue.sh run <case> -- <cmd...>     # before/after snapshot + CSV row
#   diag_residue.sh watch [timeout_s] [gb]     # poll current until <= gb GB
#   diag_residue.sh leftover                   # just dump leftover procs/fds
#
# CSV (appended to $MOE_RESIDUE_CSV, default /tmp/moe_residue_diag.csv):
#   case,before_current_GB,peak_GB,after_current_GB,oom_kill_delta,
#   leftover_process,memfd_leftover,conclusion
# ============================================================================
set -uo pipefail

CG=/sys/fs/cgroup
CSV="${MOE_RESIDUE_CSV:-/tmp/moe_residue_diag.csv}"
# Processes that indicate a live/leftover MoE run. utilize.py and ipykernel
# are deliberately EXCLUDED — they are known long-lived background tenants.
MOE_PAT='torchrun|moe_infinity|/archer|qwen|distributed_engine|m1_smoke|m2_forward|m3_forward|m3_verify|m4_long_gen|entry\.py'

gb() { awk -v b="${1:--1}" 'BEGIN{ if(b+0<0){print "?"} else printf "%.2f", b/1073741824 }'; }
cg_field()  { cat "$CG/$1" 2>/dev/null || echo -1; }
cg_current(){ cg_field memory.current; }
cg_peak()   { cg_field memory.peak; }
cg_max()    { cg_field memory.max; }
oom_kill()  { awk '/^oom_kill /{print $2}' "$CG/memory.events" 2>/dev/null || echo 0; }

leftover_procs() {
  ps -eo pid,user,rss,etime,cmd 2>/dev/null \
    | grep -iE "$MOE_PAT" | grep -v grep | grep -v diag_residue
}
memfd_count()     { find /proc/[0-9]*/fd -lname '*memfd*'   2>/dev/null | wc -l; }
moe_memfd_count() { find /proc/[0-9]*/fd -lname '*moe_numa*' 2>/dev/null | wc -l; }
sock_leftover()   { ls /tmp/moe_numa*.sock 2>/dev/null | wc -l; }
gpu_procs()       { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null; }

rank_pinned() {
  # VmLck/VmPin/VmRSS for each leftover MoE rank — what it still holds.
  local pid
  for pid in $(leftover_procs | awk '{print $1}'); do
    if [ -r "/proc/$pid/status" ]; then
      awk -v p="$pid" '/^Vm(RSS|HWM|Lck|Pin|Swap):/{printf "    pid=%s %s\n", p, $0}' "/proc/$pid/status"
    fi
  done
}

snapshot() {
  local label="${1:-?}"
  local cur peak mx ok lc moemc sk nproc
  cur=$(cg_current); peak=$(cg_peak); mx=$(cg_max); ok=$(oom_kill)
  lc=$(memfd_count); moemc=$(moe_memfd_count); sk=$(sock_leftover)
  nproc=$(leftover_procs | grep -c . || true)
  echo "===== SNAPSHOT [$label] $(date '+%F %H:%M:%S') ====="
  echo "  cgroup.current   = $(gb "$cur") GB"
  echo "  cgroup.peak      = $(gb "$peak") GB   (cumulative high-water)"
  echo "  cgroup.max       = $(gb "$mx") GB"
  echo "  oom_kill (total) = $ok"
  echo "  MoE leftover procs=$nproc  memfd(total)=$lc  moe_numa memfd=$moemc  sockets=$sk"
  if [ "${nproc:-0}" -gt 0 ]; then
    echo "  --- leftover MoE procs ---"; leftover_procs | sed 's/^/    /'
    echo "  --- their locked/pinned mem ---"; rank_pinned
  fi
  echo "  --- GPU compute procs ---"; gpu_procs | sed 's/^/    /' || true
  echo "==========================================================="
}

watch_drop() {
  local timeout="${1:-180}" target="${2:-12}"
  local deadline=$(( $(date +%s) + timeout )) cur
  echo "[diag] watching cgroup.current until <= ${target}GB (timeout ${timeout}s)"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    cur=$(cg_current)
    echo "  $(date '+%H:%M:%S') current=$(gb "$cur") GB"
    if awk -v c="$cur" -v t="$target" 'BEGIN{exit !(c/1073741824 <= t)}'; then
      echo "[diag] OK — dropped to <= ${target}GB (clean teardown)"; return 0
    fi
    sleep 2
  done
  echo "[diag] TIMEOUT — current still > ${target}GB after ${timeout}s. RESIDUE SUSPECTED."
  snapshot "watch-timeout"
  return 1
}

run_wrap() {
  local case_label="${1:?case label required}"; shift
  [ "${1:-}" = "--" ] && shift
  [ "$#" -gt 0 ] || { echo "no command after --"; exit 2; }

  local before_cur before_ok
  before_cur=$(cg_current); before_ok=$(oom_kill)
  snapshot "BEFORE:$case_label"

  # Observed-peak poller (memory.peak is read-only on this kernel, so poll
  # current at 2s and keep the max).
  local peakfile; peakfile=$(mktemp)
  echo "$before_cur" > "$peakfile"
  ( while :; do c=$(cg_current); p=$(cat "$peakfile" 2>/dev/null || echo 0)
      [ "${c:-0}" -gt "${p:-0}" ] && echo "$c" > "$peakfile"; sleep 2; done ) &
  local poller=$!

  echo "[diag] >>> running: $*"
  "$@"; local rc=$?
  echo "[diag] <<< command exited rc=$rc"

  kill "$poller" 2>/dev/null; wait "$poller" 2>/dev/null || true
  local observed_peak; observed_peak=$(cat "$peakfile" 2>/dev/null || echo "$before_cur"); rm -f "$peakfile"

  sleep 3   # let the exit settle before measuring residue
  snapshot "AFTER:$case_label (rc=$rc)"

  local after_cur after_ok nproc moemc ood concl
  after_cur=$(cg_current); after_ok=$(oom_kill)
  nproc=$(leftover_procs | grep -c . || true); moemc=$(moe_memfd_count)
  ood=$(( after_ok - before_ok ))
  if [ "${nproc:-0}" -gt 0 ] || [ "${moemc:-0}" -gt 0 ]; then
    concl="RESIDUE (procs=$nproc memfd=$moemc)"
  elif awk -v a="$after_cur" -v b="$before_cur" 'BEGIN{exit !((a-b)/1073741824 > 20)}'; then
    concl="current did NOT return to baseline (+$(gb $((after_cur-before_cur)))GB)"
  else
    concl="clean (returned to baseline)"
  fi

  [ -f "$CSV" ] || echo "case,before_current_GB,peak_GB,after_current_GB,oom_kill_delta,leftover_process,memfd_leftover,conclusion" > "$CSV"
  echo "$case_label,$(gb "$before_cur"),$(gb "$observed_peak"),$(gb "$after_cur"),$ood,$nproc,$moemc,$concl" >> "$CSV"
  echo "[diag] CSV row → $CSV"
  echo "[diag] conclusion: $concl"
  return "$rc"
}

cmd="${1:-snapshot}"; shift || true
case "$cmd" in
  snapshot) snapshot "${1:-manual}" ;;
  leftover) leftover_procs | sed 's/^/  /'; rank_pinned; echo "memfd(total)=$(memfd_count) moe_numa=$(moe_memfd_count) sockets=$(sock_leftover)" ;;
  watch)    watch_drop "${1:-180}" "${2:-12}" ;;
  run)      run_wrap "$@" ;;
  *) echo "usage: $0 {snapshot <label>|run <case> -- <cmd...>|watch [timeout_s] [gb]|leftover}"; exit 2 ;;
esac
