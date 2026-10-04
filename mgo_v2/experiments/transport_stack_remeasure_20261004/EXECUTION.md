# Execution — c60 / substitution-OFF transport stack

Owner update: the follow-up is **cache 60%, substitution OFF**. This supersedes
the earlier cache30/substitution-ON transport-stack wording.

## Frozen workload

- Cell family: R = MATH, R4, local batch 64, decode256, Gate eviction.
- Physical GPUs: 0,1,4,5.
- Cache ratio: **0.60**.
- Expert substitution: **OFF**.
- Policies: BR and CA.
- H2D: CoSLoT-style pinned two-buffer staging for all timed cells.
- Transport variants: current, coslot, coslot-active.

## PLAN handling

Do not request a PLAN path from the owner.

For BR and CA independently:
1. search only for a validated **c60/s0** PLAN;
2. if absent, generate current/pageable c60/s0 PLAN outside timing;
3. run validate_env_offload_plan.py;
4. freeze that PLAN;
5. reuse it for current/coslot/coslot-active.

The launcher verifies both cache_ratio=0.60 and substitution=false from rank0
receipt before allowing PLAN reuse.

## Timing protocol

Per policy/transport/environment:
- COMPILE x1;
- MEASURE x3;
- COUNTERS x1.

Run Env1 first. Env2 is allowed only if every Env1 E2E and TPOT
(max-min)/median spread is <=5%. No additional noise repetitions.

All pinned measurements use fixed 24-CPU affinity and ready/GO boundaries;
resource monitoring stays outside MEASURE.

## Why c60/s0

Completed CPU replay for the matching substitution-OFF condition:
- c30 BR: exact hit ~59.9%, residual miss ~40.1%, H2D ~7.16 TiB;
- c60 BR: exact hit ~86.5%, residual miss ~13.5%, H2D ~3.26 TiB;
- BR peer bytes remain ~104.5 GiB.

Thus c60 sharply lowers fetch pressure without substitution. Note that CA's
peer-byte advantage is smaller at c60 than c30, so no TPOT improvement is
assumed in advance; this experiment tests whether reduced H2D masking or
reduced placement headroom dominates.

## Driver

Use:
scripts/run_transport_stack_packet.py

Raw artifacts remain under:
/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003/

Orchestration receipts/results are under:
/home/hwlee/mgo-results/transport_stack_remeasure_20261004/

Commit and push checkpoints to:
codex/coslot-comm-remeasure-20261004

On failure, stop the matrix, preserve the receipt, clean experiment processes,
and restore resident model workers. No further experiment follows
automatically.
