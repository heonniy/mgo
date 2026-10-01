#!/usr/bin/env bash
# Robust residue cleanup for coslot 235B runs.  Use BEFORE every launch AND on
# kill.  Why residue persisted before:
#   1. orphaned ranks: killing only the launcher/torchrun leaves the 4 rank
#      processes (reparented to init) alive; the NUMA shm memfd is shared by
#      leader+follower via SCM_RIGHTS, so it is freed ONLY when EVERY mapper
#      dies → one survivor keeps the whole 888GB pinned.
#   2. pattern gap: sweep_common only matched torchrun|m[1-4]_*, not the
#      coslot_*.py rank scripts.
#   3. lazy reclaim: even after all procs die, the memfd + cudaHostRegister
#      pinned pages are reclaimed asynchronously (~30-60s) — must WAIT+verify.
#
# This kills the FULL tree (2-pass for reparented children), removes shm
# sockets/locks, then polls until shared-mem returns to baseline AND all GPUs
# are ~empty (or warns on timeout).
set -u

PATS=('coslot_owner_bench\.py' 'coslot_routing_trace\.py' 'm[1-4]_[a-z_]*\.py' 'scripts/numa_wrap' '[t]orchrun')
BASE_GB=${COSLOT_SHM_BASE_GB:-500}   # host baseline shm (~327-415G) + margin
TIMEOUT_S=${COSLOT_CLEANUP_TIMEOUT_S:-150}

echo "[cleanup] $(date -u +%FT%TZ) killing coslot/torchrun tree..."
for pass in 1 2; do
  for pat in "${PATS[@]}"; do
    pgrep -f "$pat" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  done
  sleep 2
done
rm -f /tmp/moe_numa*.sock /tmp/*.init.lock 2>/dev/null || true
sync 2>/dev/null || true

# poll until MY residue is reclaimed.  NOTE: `free` "shared" is host-wide and
# includes other tenants on this shared box (fluctuates 327-579G independent of
# us), so it is NOT a valid drain signal — gate on MY cgroup memory.current
# (one shm region is 444GB, so <60G ⇒ no shm residue of mine) + procs + GPU.
cg_gb() { awk '{printf "%.0f",$1/1073741824}' /sys/fs/cgroup/memory.current 2>/dev/null || echo 0; }
MY_GB=${COSLOT_MY_RESIDUE_GB:-60}
n=$((TIMEOUT_S / 3))
for i in $(seq 1 "$n"); do
  procs=$(pgrep -f 'coslot_owner_bench\.py|coslot_routing_trace\.py|[t]orchrun|scripts/numa_wrap' 2>/dev/null | wc -l)
  cg=$(cg_gb)
  gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  if [ "$procs" -eq 0 ] && [ "$cg" -le "$MY_GB" ] && [ "$gpu_used" -le 3000 ]; then
    echo "[cleanup] OK drained: procs=0 cgroup=${cg}G gpu_used=${gpu_used}MiB host_shared=$(free -g|awk 'NR==2{print $6}')G (${i}x3s)"
    exit 0
  fi
  sleep 3
done
echo "[cleanup] WARN not fully drained after ${TIMEOUT_S}s: procs=$(pgrep -f 'coslot_owner_bench\.py|[t]orchrun'|wc -l) cgroup=$(cg_gb)G gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|awk '{s+=$1}END{print s+0}')MiB"
exit 0
