#!/usr/bin/env bash
# cleanup_prev_run.sh — safely reclaim a previous (possibly crashed) MoE run's
# resources before starting a new one, then BLOCK until cgroup.current actually
# drops. Refuses (exit 1) if it cannot, so the caller never launches on top of
# residue and triggers an OOM-kill.
#
# Safety: PID-file (process-group) based + a TARGETED cmdline scan that excludes
# this script's own shell and parent. It NEVER uses a broad `pkill -f` that could
# take down the calling shell.
#
# Usage:  scripts/cleanup_prev_run.sh
# Env:    MOE_RUN_PIDFILE (default /tmp/moe_run.pgid) — pgid written by the launcher
#         CLEANUP_WAIT_S   (default 180)  — how long to wait for reclaim
#         CLEANUP_TARGET_GB(default 20)   — cgroup.current target to consider clean
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIAG="$REPO/scripts/diag_residue.sh"
PIDFILE="${MOE_RUN_PIDFILE:-/tmp/moe_run.pgid}"
WAIT_S="${CLEANUP_WAIT_S:-180}"
TARGET_GB="${CLEANUP_TARGET_GB:-20}"
SELF=$$
PARENT=${PPID:-0}
# Rank processes from any milestone driver (m1_smoke/m2_forward/.../m4_long_gen)
# OR a main_exp runner (run_ours_ep / run_upstream_naive), plus a torchrun that
# launched one. Deliberately specific (anchored to known runner filenames so a
# broad `pkill -f` never takes down an unrelated process or this shell).
PAT='(scripts/m[0-9][^ ]*|run_ours_ep|run_upstream_naive)\.py'

term_then_kill_group() {  # $1 = pgid
  local pgid="$1"
  kill -0 "-$pgid" 2>/dev/null || return 0
  echo "[cleanup] leftover process-group pgid=$pgid → SIGTERM"
  kill -TERM "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "-$pgid" 2>/dev/null || return 0; sleep 1; done
  echo "[cleanup] pgid=$pgid still alive after 15s → SIGKILL"
  kill -KILL "-$pgid" 2>/dev/null || true
}

# 1) PID-file route: kill the recorded process group as a unit.
if [ -f "$PIDFILE" ]; then
  pgid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [ -n "${pgid:-}" ] && [[ "$pgid" =~ ^[0-9]+$ ]] && term_then_kill_group "$pgid"
  rm -f "$PIDFILE"
fi

# 2) Targeted orphan scan (no pidfile, e.g. crash in a previous session).
#    Match the rank/torchrun cmdline, but NEVER this script or its parent.
mapfile -t orphans < <(pgrep -f "$PAT" 2>/dev/null || true)
for p in "${orphans[@]:-}"; do
  [ -z "${p:-}" ] && continue
  [ "$p" = "$SELF" ] && continue
  [ "$p" = "$PARENT" ] && continue
  # double-check this pid's cmdline really is a rank/torchrun, not us
  cmd="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)"
  case "$cmd" in
    *cleanup_prev_run.sh*|*diag_residue.sh*) continue ;;
  esac
  echo "[cleanup] orphan rank pid=$p → SIGKILL  ($(echo "$cmd" | cut -c1-80))"
  kill -KILL "$p" 2>/dev/null || true
done

# 3) Stale leader sockets (leader os._exit(0) leaves these; next leader unlinks
#    on bind, but remove proactively so a stale path never confuses diagnosis).
rm -f /tmp/moe_numa*.sock 2>/dev/null || true

# 4) Block until the kernel has actually reclaimed the freed pinned/memfd pages.
echo "[cleanup] waiting for cgroup.current to drop (<= ${TARGET_GB}GB, ${WAIT_S}s)"
if ! bash "$DIAG" watch "$WAIT_S" "$TARGET_GB"; then
  echo "[cleanup] FATAL: cgroup.current did not drop — residue is stuck (a rank" >&2
  echo "          may be wedged in an NCCL collective / D-state). Inspect with:" >&2
  echo "            bash $DIAG leftover" >&2
  echo "          and kill the holder manually before retrying." >&2
  exit 1
fi
echo "[cleanup] clean — safe to launch."
