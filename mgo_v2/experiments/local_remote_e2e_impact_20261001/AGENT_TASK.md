# AGENT TASK — Local/remote communication and E2E impact

Read:
1. README.md
2. PLAN.md
3. matrix.json
4. ../SERVER_VALIDATION_RESULTS.md
5. ../AGENTS.md

## Stage A

Use the physical mgo_v2 runtime and actual Qwen3 event shapes.

- Same tokens/experts across locality modes.
- Same hard quotas.
- No expert H2D in timed ranges.
- >=20 timed iterations after warmup.
- Publish achieved local/remote pair fractions and actual MoE/NCCL timings.

## Stage B

Run real offloading without profiler instrumentation for primary timing.

Cells:
- R4/B8/cache30;
- R8/B8/cache30;
- R8/B4/cache30 control.

Policies:
- Random;
- Hungarian current;
- Hungarian same+path.

Freeze support64, alpha=.25, path support64, eta=.5.

Use 64 fixed decode steps, five repeats. If resources remain, confirm Random vs same+path at 128 steps.

Only after timing is complete, profile one Random and one same+path repeat.

## Required outputs

Create in this folder:

- EXECUTION_BINDING.md
- measurement_manifest.json
- local_remote_sensitivity.csv/json
- e2e_repeats.csv/json
- profile_summary.csv/json
- RESULTS.md
- figure-support CSVs
- validation/provenance/hash receipts

Do not commit giant raw Nsight traces.

Stop for owner review. No timing-based retuning.
