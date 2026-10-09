# Stage-I execution breakdown

Primary performance comes only from the three unprofiled repeats in the reference cohort. Granular runs and one-forward Torch CPU/CUDA traces are separate intrusive diagnostics. All four ranks retain the reference tokens and final cache state exactly.

| Arm | Primary median TPOT (s) | H2D + rendezvous (s/token, granular diagnostic) | Compute + rendezvous (s/token, granular diagnostic) |
|---|---:|---:|---:|
| R-NEAR | 1.943698 | 1.077150 | 0.625818 |
| G-NEAR | 1.930163 | 1.059801 | 0.638810 |

## Metadata and PLAN

The earlier ~0.25 s/token residual includes attention/router and must not be called metadata overhead. BREAKDOWN.json now separates the global MoE-entry frontier from metadata/PLAN completion. This is a frontier partition, not a sum of CUDA kernels. CPU calls are nested diagnostic spans and must not be summed with phase times. The active profiler step is excluded from the steady CPU-call summary.

| Arm | Metadata call max-rank mean (ms/layer) | C++ controller call | CPU layout | Index pack |
|---|---:|---:|---:|---:|
| R-NEAR | 2.3355 | 0.2357 | 0.2499 | 0.5891 |
| G-NEAR | 2.3534 | 0.2352 | 0.2545 | 0.5920 |

## Expert compute

The native C++ executor calls separate gate/up/down GEMMs for every expert. Moving the loop to C++ removed per-expert Python crossings but does not remove ATen/cuBLAS launches, per-expert gather/weight operations or small-M inefficiency. Kernel reports correlate CUDA launches inside each executor span to their GPU kernels and report the union of actual kernel-active intervals. CUDA stream windows include host submission gaps; they are not GPU-active time. Profiler CPU overhead prevents a production-speed claim from this one early decode step.

A separate paired grouped/native probe uses identical packets, cache weights and routing weights, with all demand H2D already complete. Only the original native output advances the model, preserving its entire routing/cache trajectory. Its timing is compute-only, not serving TPOT.

Ready-First is a different schedule: compute resident/completed experts while other mandatory H2D remains in flight. The serialized mode neutralizes that opportunity because all ranks finish H2D before compute. A separate Ready-First mode must still finish all metadata/forward communication before any H2D and prevent return communication until all H2D and compute are complete. Grouping ready waves trades earlier compute against more launch batches.

Potential overhead work: native index/layout preparation; reuse metadata/header storage; replace pickle-based checksum preparation with a deterministic native digest while retaining per-event PLAN validation. Removing checks or phase barriers would change the experiment and is not included in these primary results.
