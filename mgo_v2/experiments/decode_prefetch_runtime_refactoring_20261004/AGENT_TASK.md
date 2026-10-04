# AGENT TASK — decode prefetch runtime refactoring

This branch is PLAN-FIRST. Do not jump directly to the final integrated runtime.

Read in order:
1. README.md
2. INVARIANTS.md
3. PLAN.md
4. MILESTONES.md
5. MEASUREMENT.md
6. matrix.json

## Mandatory execution discipline

- Implement milestones sequentially.
- Commit each milestone independently.
- Never combine cache-role refactoring, metadata refactoring, H2D scheduler, and
  communicator rewrite in one commit.
- Every milestone must include tests and a validation receipt.
- Stop on any token/cache/plan drift.
- Do not run long GPU experiments automatically after implementation.
- Replication is not part of this branch's active method.

## Main code references

Inspect before modifying:
- current `mgo_v2/mgo_v2/cache.py`;
- current `mgo_v2/mgo_v2/controller.py`;
- current `mgo_v2/mgo_v2/pinned_h2d.py`;
- current `mgo_v2/mgo_v2/communicator.py`;
- current `mgo_v2/examples/env_offload_worker.py`;
- main branch `MoE-Infinity-EP-coslot/.../demand_collector.py`;
- main branch `MoE-Infinity-EP-coslot/.../slot_cache.py`;
- main branch `MoE-Infinity-EP-coslot/.../nvlink_router.py`;
- main branch Archer prefetch task scheduler;
- main branch MoE-Infinity expert tracer/predictor/prefetcher.

## First implementation checkpoint

Do only M0-M3 first:
- freeze baseline;
- baseline phase instrumentation;
- transition calibration;
- CPU predictor/P/BR-CA-LA characterization.

Report results and stop for review before changing physical cache memory layout.

## Second checkpoint

M4-M7:
- C+P arena;
- decode boundary;
- compact metadata;
- split controller.

Stop and validate.

## Third checkpoint

M8-M12:
- priority async H2D;
- prefetch rank placement;
- ready-first execution;
- 2-round A2A;
- integrated overlap.

Stop and validate.

## Final checkpoint

M13-M15:
- physical trigger/P sweep;
- phase exposure;
- BR/CA/LA final comparison.

Every performance claim must be reproduced without profiler instrumentation.
