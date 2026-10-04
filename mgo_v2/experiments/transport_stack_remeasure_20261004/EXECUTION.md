# Execution of ac6d6bc

Owner authorized the latest branch on 2026-10-04. This supersedes the earlier
missing-R-PLAN block. R means MATH, R4, local batch 64, decode256, cache30,
substitution ON, physical GPUs 0,1,4,5 as specified by the frozen matrix.

Create any missing current/pageable BR and CA reference PLANs outside timing,
validate route/action/cache hashes, commit each, and freeze them for all runs.
Use the owner's current/pinned, coslot/pinned and coslot-active/pinned code;
two expert-sized pinned buffers per rank. No full-model pinned allocation.
Six CPU layout/staging/peer tests pass; GPU parity gates remain mandatory.

Each policy/transport/environment uses COMPILE x1, MEASURE x3, COUNTERS x1.
This latest explicit protocol overrides the older single-shot screen. Env1
has 18 timed runs; a stable Env1 permits the same 18 Env2 runs. Use the
previously validated stability threshold: (max-min)/median <=5% for both
E2E and TPOT in every condition. No extra noise repeats. Also report when
BR/CA median differences fall within within-policy ranges; a stable harness
does not prove a policy gain. Env2 is SHM stress, not a different PCIe host.

All pinned measurements use fixed 24-CPU affinity and ready/GO boundaries;
resource scans stay outside MEASURE. Existing host/GPU memory and thermal
guards remain active. The prior historical STOP marker is archived when this
explicit restart begins. Completed historical artifacts are retained.

Driver: scripts/run_transport_stack_packet.py. Raw matrix artifacts remain
under /home/hwlee/mgo-results/env_e2e_tpot_offload_20261003/ using transport and
pinned suffixes; orchestration logs are in
/home/hwlee/mgo-results/transport_stack_remeasure_20261004/.
Commits and pushes target codex/coslot-comm-remeasure-20261004. On any failure,
stop the matrix and preserve the failed receipt. Restore resident-model load
workers after cleanup. No further experiment follows automatically.
