# Rank-demand oracle and GPU critical-path study

Status: PASS. Plan `dccc93c`, with owner-authorized scope reduction when `scope_amendment.json` is present; physical GPUs 0,1,4,5 only. Exactly 30 unprofiled full generations across three local batches; B4/B8/B16 repetitions per policy: 6/2/2. One prefill plus 64 decode forwards; cache/history reset for every generation. Solver planning time is excluded from frozen-oracle timing.

## Primary measurements

| Local B | Policy | E2E median [min, max], s | TPOT median [min, max], s | Decode sum busiest-rank rows | H2D GiB | Remote pairs |
|---:|---|---:|---:|---:|---:|---:|
| 4 | P0 | 91.356 [90.866, 135.338] | 1.3575 [1.3405, 2.0423] | 113,343 | 417.26 | 272,230 |
| 4 | P1 | 97.005 [91.883, 117.490] | 1.4362 [1.3687, 1.7476] | 129,725 | 425.13 | 239,421 |
| 4 | O0 | 91.096 [89.807, 137.594] | 1.3537 [1.3354, 2.0732] | 96,818 | 413.30 | 281,454 |
| 8 | P0 | 131.688 [125.629, 137.746] | 1.9293 [1.8283, 2.0302] | 236,958 | 615.01 | 561,980 |
| 8 | P1 | 136.467 [121.613, 151.321] | 2.0194 [1.7794, 2.2594] | 293,814 | 579.73 | 465,948 |
| 8 | O0 | 121.121 [120.167, 122.075] | 1.7889 [1.7806, 1.7971] | 188,883 | 596.07 | 584,196 |
| 16 | P0 | 224.166 [166.837, 281.495] | 3.3273 [2.4374, 4.2173] | 481,605 | 831.74 | 1,114,684 |
| 16 | P1 | 185.180 [175.456, 194.903] | 2.6763 [2.5082, 2.8444] | 609,373 | 842.58 | 935,948 |
| 16 | O0 | 183.934 [168.267, 199.601] | 2.7244 [2.4847, 2.9641] | 369,240 | 854.16 | 1,155,405 |

Timing dispersion is material: B16/P0 E2E ranges from 166.837 to 281.495 s (68.7% spread relative to the faster run). With the reduced repeat count and shared-host interference, this packet does not establish a stable E2E advantage from these median differences alone.

B8 median maximum-rank controller time: P0 85.801 s of 131.688 s E2E; P1 91.413 s of 136.467 s E2E; O0 80.629 s of 121.121 s E2E. The separately authorized controller repair addresses this host-side cost.

P0 = Balanced Random; P1 = Hungarian-current; O0 = exact rank-demand oracle replay. O0 is an exact per-event load-headroom diagnostic conditional on its own cache state and effective demand; it is neither a global trajectory optimum nor an online deployable speedup. Min/max are observed ranges, not confidence intervals. All 65 forward outputs are counted even after EOS; throughput is a fixed-work token output rate.

## Oracle headroom and controls

- B4, O0 relative to P0: TPOT reduction +0.28%; E2E reduction +0.28%; decode busiest-rank row reduction +14.58%; remote-pair change +3.39%; H2D change -0.95%.
- B4, O0 relative to P1: TPOT reduction +5.75%; E2E reduction +6.09%; decode busiest-rank row reduction +25.37%; remote-pair change +17.56%; H2D change -2.78%.
- B8, O0 relative to P0: TPOT reduction +7.28%; E2E reduction +8.02%; decode busiest-rank row reduction +20.29%; remote-pair change +3.95%; H2D change -3.08%.
- B8, O0 relative to P1: TPOT reduction +11.42%; E2E reduction +11.25%; decode busiest-rank row reduction +35.71%; remote-pair change +25.38%; H2D change +2.82%.
- B16, O0 relative to P0: TPOT reduction +18.12%; E2E reduction +17.95%; decode busiest-rank row reduction +23.33%; remote-pair change +3.65%; H2D change +2.69%.
- B16, O0 relative to P1: TPOT reduction -1.80%; E2E reduction +0.67%; decode busiest-rank row reduction +39.41%; remote-pair change +23.45%; H2D change +1.37%.

Policies can change later substitution, routes and fetches. These complete model trajectories do not hold raw demand constant across policies. H2D, reloads, next-use survival and controller time are reported alongside rank load; material differences confound a pure compute-balance interpretation. Pair counts and modeled activation payload exclude NCCL protocol overhead and are not wire traffic measurements.

## Solver and validation

- B4: 200.227 s exact solve time across 3120 events; slowest event 10.630 s. Every solve is optimal; no heuristic fallback.
- B8: 542.810 s exact solve time across 3120 events; slowest event 35.725 s. Every solve is optimal; no heuristic fallback.
- B16: 1378.457 s exact solve time across 3120 events; slowest event 126.762 s. Every solve is optimal; no heuristic fallback.

All recorded O0 replays reproduce every event route/substitution/admission/victim/cache-slot hash and every full generated token on all ranks. Each policy repeats deterministically. Instrumentation on/off passes the short GPU gate; full B8 profile parity is separately required. CPU exhaustive tests verify both the optimal load and unique lexicographic tie-break.

## GPU critical path

Decode-only sums of per-event maxima across four ranks; these maxima need not occur on the same rank. Kernel union excludes idle gaps; span includes gaps and waits.

| Policy | Expert kernel union max sum (ms) | GEMM union max sum (ms) | Expert span max sum (ms) | NCCL union max sum (ms) | H2D union max sum (ms) |
|---|---:|---:|---:|---:|---:|
| P0 | 2481.286 | 772.911 | 30456.142 | 53642.705 | 3238.129 |
| P1 | 2504.513 | 776.156 | 29461.693 | 31363.549 | 3071.550 |
| O0 | 2388.018 | 742.393 | 28016.790 | 27271.792 | 3172.353 |

The profile includes native worker-thread kernels inside each enclosing MoE layer. Expert ranges include execution preparation and concatenation kernels; the GEMM-only series isolates matrix kernels. H2D copy intervals come from CUPTI, not the asynchronous CPU submission duration. GPU interval attribution must pass expected GEMM counts and exact H2D byte reconciliation. The three profiles follow all primary timing; profile wall times are excluded from speed claims.

### Answers to the planned questions

1. P1 versus P0: decode busiest-rank rows change +23.99%; measured expert-kernel max-rank time changes +0.94%. This is a descriptive physical comparison, with policy-dependent raw routes.
2. O0 versus P1: expert-kernel max-rank time changes -4.65%; versus P0, -3.76%. GEMM-only and elapsed-span alternatives are retained above.
3. At B8, O0 versus P1 changes primary median TPOT -11.42% and E2E -11.25%. These use the unprofiled repetitions listed above, not profile wall time.
4. O0 versus P1 changes B8 remote pairs +25.38% and profiled max-rank NCCL kernel union -13.05%. NCCL intervals include device-side waiting.
5. O0 versus P1 changes B8 physical H2D volume +2.82% and median maximum-rank controller time -11.80%. These concurrent changes and shared-host noise limit attributing all E2E change to compute balance.

## Resource and measurement limits

All new jobs were sequential R4 jobs. Other users continued work on GPUs 2,3,6,7 and shared CPU, RAM and storage. B4 uses all six policy orders. The reduced B8/B16 forward/reverse pair keeps P1 in the middle and does not fully balance position. Two repetitions support descriptive checks only; they do not establish small performance differences. Ordering does not remove shared-host interference. No long-horizon quality conclusion is made from this fixed-work numeric workload. B4/B8/B16 use the corresponding 16/32/64-question prefix, so cross-batch differences also change workload composition and W128 history span.
Timing retains compact CPU evidence and planned-row counting for all policies, plus frozen-demand checks for O0. Hashing, serialization and token checks run outside the generation timer. No NVTX/CUDA events/CUPTI are active in primary timing. See IMPLEMENTATION.md for exact boundaries and memory guards.
Observed minimum host availability: 1487.5 GiB; minimum selected-GPU free memory: 69,339 MiB. Initial smoke RSS observations included only the launcher process group; full planning/timing monitoring counts descendants as well.

## Reproduction and artifacts

Launcher: `scripts/run_rank_oracle_study.py` for smoke/planning/profiles and the preserved original B4 matrix; `scripts/run_rank_oracle_reduced.py` for the owner-amended B8/B16 timing matrix. The original unamended timing stage would schedule the larger matrix and must not be used to resume this reduced packet. Analysis: `scripts/analyze_rank_oracle_profiles.py`, `scripts/summarize_rank_oracle_study.py`. Raw traces remain at `/home/hwlee/mgo-results/rank_demand_oracle_20261002`; giant Nsight traces are not committed.
Inspect `e2e_repeats.csv`, `e2e_summary.csv`, `rank_load_summary.csv`, `rank_load_events.csv`, `oracle_planning_summary.csv`, `gpu_phase_profile.csv`, `gpu_critical_path.csv`, `validation.json` and hash receipts. CSVs supply figure data.

No joint communication/load policy, production controller optimization, substitution retuning, replication or migration was added.
