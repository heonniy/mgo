# Exact-only payload distribution and bounded transport crossover

**Status: PASS. Decision: USEFUL_CONTRAST_WITHIN_OBSERVED_RANGE.** One diagnostic model capture and the prescribed five-size/two-pass crossover are complete. Stop for owner review; no Stage 1 or replication work was started.

## Exact-only capture

R4 on physical GPUs 0,1,4,5; B8 per rank; Qwen3-30B-A3B-Instruct-2507 BF16; cache30; Balanced Random P0 seed 42; LRU; substitution and replication disabled. Exactly one prefill plus eight decode forwards, with no model warmup or quality evaluation. Its wall time is not used as performance evidence.

Every selected raw expert has an observed execution owner. Actual send counts were captured at `_all_to_all_varlen` for dispatch hidden states and exact-expert return outputs. All 432 event plan/cache hashes match across ranks; per-event physical cache checks pass. Independently computed raw-exact counts agree with recorded sends, and every send matrix equals the transpose of its receive matrix.

| Phase | Nonzero messages | p50 KiB | p90 KiB | p99 KiB | Maximum KiB | <=64 KiB messages | 64–256 KiB messages |
|---|---:|---:|---:|---:|---:|---:|---:|
| dispatch | 4,608 | 32 | 32 | 32 | 32 | 100.00% | 0.00% |
| combine | 4,608 | 64 | 88 | 112 | 140 | 56.29% | 43.71% |
| union | 9,216 | 32 | 80 | 108 | 140 | 78.15% | 21.85% |

Only the 384 decode layer events enter these distributions; zero and self messages are excluded. Each activation row is 2,048 BF16 elements (4,096 bytes). Combine returns one weighted row per exact expert; dispatch sends a token once per destination. All messages above 256 KiB have zero observed frequency. Metadata and NCCL protocol overhead are excluded.

## Crossover

T0 clears P2P/IB/GDR/SHM overrides; R3 sets only `NCCL_P2P_LEVEL=LOC` and `NCCL_IB_DISABLE=1`. First-size-only INFO smokes prove T0 `P2P/CUMEM` and R3 `SHM/direct/direct` on all four ranks. Timing runs have INFO disabled. No channels, protocols, batching or other transport knobs were tuned.

Pass 0 is T0→R3; pass 1 is R3→T0. Every size/mode/pass has 10 warmups and 30 timed `all_to_all_single` calls. Each sample is the maximum rank CUDA interval, including stream/launch waiting. Primary latency is the median of the two pass medians. The p90 below pools the 60 per-iteration maximum-rank intervals; separate pass medians/p90s and ratios are in the CSV. Effective bandwidth uses remote transmit payload per rank (three peers × size), excludes self/NCCL overhead, and is not aggregate fabric bandwidth.

| Per-peer size | T0 median ms | R3 median ms | R3/T0 | Pass 0 / pass 1 ratios | T0 / R3 pooled p90 ms | T0 / R3 GB/s per rank |
|---|---:|---:|---:|---:|---:|---:|
| 32 KiB | 0.047624 | 0.120104 | 2.522x | 2.139 / 2.912 | 0.055901 / 0.155101 | 2.064 / 0.818 |
| 64 KiB | 0.072256 | 0.064440 | 0.892x | 1.503 / 0.578 | 0.128150 / 0.077808 | 2.721 / 3.051 |
| 256 KiB | 0.056888 | 0.097512 | 1.714x | 2.119 / 1.406 | 0.074064 / 0.107658 | 13.824 / 8.065 |
| 1024 KiB | 0.056152 | 0.214312 | 3.817x | 3.808 / 3.825 | 0.082538 / 0.220960 | 56.022 / 14.678 |
| 4096 KiB | 0.095088 | 0.697080 | 7.331x | 7.338 / 7.324 | 0.100762 / 0.712218 | 132.329 / 18.051 |

## Decision and limits

The predeclared >=1.5x slowdown criterion is met within the observed exact-only p99 range at: 32 KiB. This supports the H100 synthetic contrast for owner consideration; it does not authorize Stage 1 in this follow-up.

Exactly 32 KiB accounts for 33.18% of measured messages and 22.71% of activation bytes. At or above that size, 80.69% of observed messages and 89.67% of observed activation bytes lie in messages at or above that size. This is distribution coverage, not interpolation of an unmeasured latency curve.

The 32 KiB slowdown exceeds 1.5x in both passes (2.139x and 2.912x). At 64 KiB, however, the ratios reverse from 1.503x to 0.578x; the aggregate ratio is 0.892x. Thus the useful contrast is size-specific, and no single monotonic crossover or general slowdown throughout the 32–112 KiB range is established. Larger-message ratios are not used to justify this decision because those sizes exceed the observed maximum.

These values should not be compared numerically with the earlier two-operation 32/64 KiB proxy: this follow-up times one all-to-all at each fixed size and uses two nearby counter-ordered passes.

This is a short single-capture payload sample under P0/LRU, not a universal bound for other batches, prefill or residency policies. The microbenchmark uses equal payloads for all rank pairs; real MoE pairs vary. Two counter-ordered passes reduce temporal bias but do not establish broad timing repeatability. Do not infer that SHM is intrinsically faster or slower at all sizes.

## Evidence and resource use

Capture peak process-tree RSS was 226.9 GiB and minimum host availability 1613.0 GiB. No OOM or memory-guard stop occurred. Jobs were sequential on GPUs 0,1,4,5; other GPU workloads were untouched.

Full capture records contain raw selections, token origins, exact owner maps, effective routes, actual send/receive counts and plan/cache hashes. They remain under `/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/exact_payload_20261003`. Raw file/source hashes, config, validation and microbenchmark provenance are in `exact_payload_capture_summary.json`; decode count matrices are retained in `payload_distribution.json`. The two compact INFO logs retain original path-verification lines; full logs are external.

The previous blocked existing-trace audit remains at `0c09fae`; the new authorized exact-only capture resolves the missing-owner issue. This packet ends here for owner review.
