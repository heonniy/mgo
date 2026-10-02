# Controller overhead repair: reduced B8 confirmation

Status: PASS. This is a separate follow-up to the completed oracle packet. At the owner’s request, the matrix was reduced to one fresh fixed-work generation per policy/controller (9 total), full captured-route differential validation and two short posthoc GPU profiles. No B4/B16 controller expansion was run.

## Fresh matched physical measurements

| Policy | Controller | E2E s | TPOT s | Max-rank controller s | Controller / E2E | Controller reduction | TPOT reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | C0 | 142.122 | 2.0753 | 90.751 | 63.9% | +0.0% | +0.0% |
| P1 | C0 | 205.113 | 3.0089 | 145.518 | 70.9% | +0.0% | +0.0% |
| O0 | C0 | 152.779 | 2.2444 | 96.128 | 62.9% | +0.0% | +0.0% |
| P0 | C1 | 70.359 | 0.9806 | 20.808 | 29.6% | +77.1% | +52.7% |
| P1 | C1 | 69.870 | 1.0126 | 24.001 | 34.4% | +83.5% | +66.3% |
| O0 | C1 | 74.667 | 1.0715 | 25.698 | 34.4% | +73.3% | +52.3% |
| P0 | C2 | 61.714 | 0.8641 | 20.885 | 33.8% | +77.0% | +58.4% |
| P1 | C2 | 61.295 | 0.8636 | 21.045 | 34.3% | +85.5% | +71.3% |
| O0 | C2 | 73.800 | 1.0679 | 22.556 | 30.6% | +76.5% | +52.4% |

C0 is the original controller, C1 is local indexed/vectorized Coverage, C2 uses the C1 planner on rank 0 plus a compact decision broadcast. P0/P1/O0 preserve their original policy trajectories. C0 controls here are fresh and do not replace the primary oracle packet.

**Single-sample descriptive measurements:** each cell has one sample; no confidence interval, stable speedup estimate or inference about small differences is justified. C0, C1 and C2 ran in that order on a shared host. The reduced matrix does not counterbalance controller order. All cells perform one prefill plus 64 decode forwards, including outputs after EOS.

## What changed and why

C1 keeps the original controller flow and admission/substitution implementations. Its cache maintains numeric slot/key/last-use views and per-layer resident sets; successful placement/eviction updates Coverage counts incrementally. Gate history uses the original float64 accumulation and float32 rounding. Array midranks retain exact ties and the original score/last-use/expert-key tie-break. Full consistency scans are gated in the primary path and exercised offline. Candidate visits are unchanged; repeated Python object construction and scoring/sorting overhead is reduced.

C2 retains all-rank routing collection. Rank 0 computes the C1 plan and sends fixed-width integer decisions; followers reconstruct effective weights in the original summation order and apply the exact cache operations. There is no second independent residency authority, replication or migration. Follower random-generator state is unused; planner failover is not implemented.

## CPU mechanism diagnostics

These are instrumented single-process replays of each fresh C0 physical raw trace, not production speed measurements. Serialization, full state comparison and offline consistency checks occur outside component timers. C0 and C1 see exactly the same raw routes and decisions.

| Policy | Controller | Total CPU controller s | Coverage rank/victim s | Candidate visits | Cache key items materialized | Full resident-set builds |
|---|---|---:|---:|---:|---:|---:|
| P0 | C0 | 90.684 | 69.906 | 30,646,150 | 31,392,509 | 68,131 |
| P0 | C1 | 20.080 | 9.788 | 30,646,150 | 0 | 0 |
| P1 | C0 | 83.550 | 63.421 | 28,866,280 | 29,543,108 | 64,117 |
| P1 | C1 | 19.214 | 8.925 | 28,866,280 | 0 | 0 |
| O0 | C0 | 82.196 | 62.958 | 29,686,738 | 30,399,610 | 65,976 |
| O0 | C1 | 18.192 | 8.790 | 29,686,738 | 0 | 0 |

## Interpretation of the reduced comparison

The measured bottleneck is repeated Coverage victim evaluation, not the Hungarian solver. In the P0 trace alone, 68,131 victim choices inspect 30,646,150 candidates (about 450 per choice). C0 also materializes 31,392,509 cache-key items and 68,131 full resident sets. C1 keeps every candidate visit and victim decision while removing those repeated full constructions and using incremental Coverage counts and array ranking. The separate CPU component replay supports this mechanism independently of the physical timing samples.

All three physical C1 samples reduce controller time substantially. This is consistent with host planning contributing to the previous critical path. It does not establish a stable end-to-end speedup magnitude: the fresh C0/P1 controller times vary from 87.844 to 145.518 seconds across ranks, and its 205.113-second generation is much slower than the original P1 B8 control samples. Shared-host scheduling variation is visible in the baseline itself.

C2 has lower observed E2E than C1 for P0/P1 (70.359 to 61.714 seconds and 69.870 to 61.295 seconds), but O0 is nearly unchanged (74.667 to 73.800 seconds). P0 maximum-rank controller time is also essentially unchanged between C1 and C2. These single samples do not establish a universal advantage for single-planner broadcast. Keep C1 and C2 separately selectable; do not infer that removing four concurrent copies of planning should provide a fourfold latency gain.

This overhead-only study supplies no new stable ranking of P0/P1/O0 placement policies. The original oracle packet remains the placement evidence. No additional repetitions or B4/B16 controller expansion are needed under the owner-reduced confirmation scope.

Detailed exclusive history, substitution, admission, Coverage synchronization and cache components are retained in `cpu_components_*.csv` and `cpu_component_summary.json`.

## Planner transport and device work

C2 broadcasts 4,656 payload bytes per layer event. The logical payload and modeled planner-to-peer bytes are recorded separately from expert H2D traffic; they exclude NCCL protocol overhead and are not measured wire bytes.

The short P1 profiles cover one prefill plus eight decode forwards after primary timing. They compare C1/C2 against the same prefix of the original C0 profile. Per-event/rank expert rows, GEMM counts and physical expert-fetch bytes matched exactly. Full 65-forward runtime fetch/cache/route counters also match for all three policies. The short profile does not characterize late-decode timing or all policies.

| Rank | Planner compute s | Encode s | Broadcast/copies/wait s | Apply s |
|---:|---:|---:|---:|---:|
| 0 | 2.7160 | 0.0506 | 0.1037 | 0.0000 |
| 1 | 0.0000 | 0.0000 | 2.8997 | 0.6220 |
| 2 | 0.0000 | 0.0000 | 2.8988 | 0.6284 |
| 3 | 0.0000 | 0.0000 | 2.9023 | 0.6377 |

Follower broadcast time includes waiting for rank 0 to finish planning. It is not a pure network-latency estimate. CPU transport spans include required copies and synchronization; actual NCCL GPU intervals and payload-copy reconciliation are retained in the profile artifacts.

GPU durations below sum the maximum rank interval-union duration at each decode layer event over the eight profiled decode forwards. NCCL includes the added C2 planner broadcast; it also includes GPU waiting and is not pure communication service time.

| Controller | Expert GPU ms | GEMM ms | NCCL ms | Expert H2D ms |
|---|---:|---:|---:|---:|
| C0 | 228.521 | 66.968 | 2651.103 | 201.800 |
| C1 | 228.672 | 66.898 | 4315.191 | 201.363 |
| C2 | 228.785 | 66.907 | 3436.130 | 201.671 |

## Correctness and resources

All 39 CPU tests passed (the 32 existing tests and seven optimization/codec tests); see `cpu_tests.txt`.

All 9,360 captured CPU events match complete C0/C1/follower plans, effective-route weights, cache owners/slots/timestamps and rolling history. All nine physical generations match the original packet event hashes and full generated tokens on every rank. Malformed payloads, rounding/ties and diagnostics parity are tested separately.

Only physical GPUs 0,1,4,5 were used, one job at a time. Minimum host availability was 1607.3 GiB; maximum process-tree RSS was 231.9 GiB; minimum selected-GPU free memory was 70,151 MiB. Guard stops: 0. Shared-host interference remains a timing limitation.

## Artifacts and reproduction

Run `run_controller_overhead.py --stage C0`, `replay_controller_equivalence.py`, then stages C1, C2 and profiles, followed by `analyze_controller_profiles.py` and this summary script. Source/protocol hashes and the frozen C0 audit are in `measurement_manifest.json`. Raw receipts and traces remain under `/home/hwlee/mgo-results/controller_overhead_20261002`; large Nsight traces are not committed.

The optimized paths remain opt-in. This packet changes implementation overhead only; no placement objective, substitution rule or Coverage policy coefficient was tuned.
