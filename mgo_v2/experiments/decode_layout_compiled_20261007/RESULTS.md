# Compiled decode layout: physical result

R4/C30/localB32/global128,input512,output64, LA_CA_NEAR/H0/full-pinned; GPUs0/1/4/5. Existing prefill optimizations enabled in both arms. Two reference primaries then two compiled primaries, each arm with disjoint full warmup. Dynamic expert/cache/history reset before every primary. All samples retained; no profiler in primary timing.

## Change

Reuse the exact canonical compiled rank-partial packet builder in decode. Keep arrays through contiguous GPU index packing; omit unused expert-order return indices and duplicate selected-ID D2H. Replace per-expert full-slot scans with one compiled layer->physical-slot lookup rebuilt after current placement/promotions. No changes to metadata all-gather, Gate history, prediction, placement policy, cache authority, arithmetic/accumulation order or H2D scheduling. This is a CPU compiled metadata/layout optimization, not a GPU controller.

## Raw clean measurements (seconds)

|Mode|Repeat|TTFT|TPOT|E2E|
|---|---|---|---|---|
|reference|1|5.207891|1.111899|75.257532|
|reference|2|3.055126|1.168215|76.652691|
|compiled|1|4.677002|0.850814|58.278307|
|compiled|2|2.771624|0.854187|56.585415|

|Metric|Reference mean ± sample SD|Compiled mean ± sample SD|Observed reduction|
|---|---|---|---|
|TTFT|4.131508 ± 1.522235|3.724313 ± 1.347305|9.86%|
|TPOT|1.140057 ± 0.039822|0.852501 ± 0.002385|25.22%|
|E2E|75.955111 ± 0.986526|57.431861 ± 1.197055|24.39%|

## Verification

- CPU100 differential cases: exact packet/group ordering and indices, empty/uneven/B16/B32/B64; physical slot role swap and eviction refresh.
- GPU12096 warmup checks:3024 decode layers/rank x4, indices/group order/physical slots match old path. Candidate actually computes with the new layout in validation.
- All16384 corresponding primary generated token positions match, as do request IDs, final cache/role hashes, controller counters, H2D bytes/copies across both repeats and all4 ranks.
- Finite logits, cache consistency and no primary Torch compilation checks pass. Numba builders are exercised in warmup.
- Headline default/explicit-reference CLI assertions pass after promotion.

## Interpretation and adopted scope

TPOT mean falls1.140057->0.852501s (25.22%) in this sequential comparison. Compiled repeat relative range0.40%; reference4.94%. E2E mean75.955111->57.431861s (24.39%). TTFT has first-repeat variability in both arms; no causal TTFT improvement claim. Reference TPOT is slower than historical0.866s, so do not advertise a universal25% gain over the old main table. This is not interleaved or a component-timer attribution study. H2D work is unchanged; metadata/layout overhead was reduced without removing model work.

Headline OURS worker now enables decode_layout_fast by default. --legacy-decode-layout retains the measured reference; comparison driver explicitly chooses each arm. Other runtime consumers remain opt-in, guarded to fused BF16 rank-partial execution. Existing main-table files are not overwritten. GPU histogram/controller migration, further policy optimization and other-cell physical measurements are outside this stage. No additional run queued.
