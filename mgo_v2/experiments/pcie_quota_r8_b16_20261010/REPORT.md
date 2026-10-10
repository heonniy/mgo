# R8 PCIe-aware miss quota experiment

## Result

The PCIe lookup was implemented and measured, but **it did not improve this R8/B16 workload**. In the matched-next-token control, the simple fast-rank quota had the lowest median TPOT (0.3839 s/token), original Near was 0.3881 s/token, and the PCIe lookup was 0.3913 s/token. All three TPOT ranges overlap. Keep the existing `main_OURS` Near quota as the default; the new quotas remain explicitly selected experimental policies. The measurements do not establish a stable end-to-end winner.

The microbenchmark does establish a PCIe asymmetry. With all eight GPUs copying eight 9 MiB experts concurrently, GPUs 0–3 finished in about 2.09–2.10 ms, while GPUs 4–7 finished in about 1.59–1.61 ms. Assigning a balanced quota's extra copy to a faster rank can therefore reduce predicted latest-rank H2D completion. A PCIe-optimal *H2D* quota does not necessarily minimize model TPOT: cache evolution, routing, expert execution, collectives, and timing noise remain in the critical path.

## Workload and method

- Qwen3-30B on all eight physical GPUs, ShareGPT, C30, B16 **per GPU**, input 512, output 64; `main_OURS` native Ready-First executor, compiled prefill/decode layout, prefetch OFF.
- The 9 MiB pinned-source calibration measured each of the 255 nonempty GPU subsets twice for one simultaneous copy per rank. Full-eight 2/4/8/32-copy anchors and 16-repeat refinements supplied the larger balanced-quota cases. Raw calibration, skew flags, and GPU-event timings are in `/home/hwlee/mgo-results/pcie_quota_r8_b16_20261010/`.
- Offline enumeration produced one balanced eight-rank quota for every global miss count 0–128. `NEAR_PCIE` minimizes the calibrated maximum rank H2D duration. `NEAR_FAST` assigns extra copies to ranks ordered by measured full-eight eight-copy speed. Both feed the unchanged Near expert assignment. The compiled controller performs one table-row lookup per layer; no online subset search or Python decision is needed.
- Each policy had two unfiltered target repeats, plus exactly one third when the first two TPOT or E2E values differed by more than 2%. Each target started from an empty expert cache after warmup. All nine matched-control targets (72 rank receipts) passed runtime validation and had zero quota violations.
- Live generated text diverged across policies. The primary comparison below therefore uses the original Near's next-token IDs for **all three** policies, while still recording each policy's own argmax. This matches decode input tokens, not BF16 hidden states or expert routing bit-for-bit. The earlier live-generation measurements remain in [RESULTS.md](RESULTS.md).

## Physical results

Reported values are the median of three target repeats; brackets are the complete unfiltered range. The second target repeat in each policy had an unusually short TTFT, so the E2E ranges are particularly wide. It was retained.

| Policy | TTFT (s) | TPOT (s/token) | E2E (s) | TPS |
|---|---:|---:|---:|---:|
| Original Near | 3.102 [1.846, 3.131] | 0.3881 [0.3839, 0.3942] | 27.579 [26.032, 27.935] | 297.041 [293.254, 314.692] |
| Fast-rank balanced quota | 3.054 [1.809, 3.068] | 0.3839 [0.3831, 0.3879] | 27.239 [25.946, 27.506] | 300.748 [297.824, 315.727] |
| PCIe lookup balanced quota | 3.082 [1.823, 3.175] | 0.3913 [0.3867, 0.3956] | 27.734 [26.184, 28.100] | 295.377 [291.526, 312.869] |

The fast-rank median TPOT is 1.08% below original Near. The PCIe-lookup median is 1.94% above fast-rank. These are **observed median differences**, not demonstrated improvements or regressions: the three-repeat ranges overlap, and policy-dependent BF16 routing still differs despite common next-token input. The first matched target's actual argmax differed from the Near teacher in 107 of 8192 output positions for fast-rank and 109 for PCIe lookup; Near differed in none. The full repeat values and provenance checks are in [FROZEN_RESULTS.json](FROZEN_RESULTS.json).

## Why the lookup did not yield a model gain

The three policies had almost the same amount of compulsory PCIe work: 211,249, 211,346, and 211,465 fetched experts, respectively, over 64 decode steps × 48 MoE layers (about 68.8 global misses per layer-step). Every fetched expert is 9 MiB, so each run moved roughly 1.81 TiB in total H2D expert data. Changing quota mainly moved copies between ranks, rather than eliminating copies.

| Physical rank | Original Near copies | Fast-rank copies | PCIe lookup copies |
|---:|---:|---:|---:|
| 0 | 27,714 | 25,121 | 25,114 |
| 1 | 27,355 | 25,491 | 25,504 |
| 2 | 26,960 | 25,846 | 25,875 |
| 3 | 26,593 | 26,218 | 26,227 |
| 4 | 26,230 | 27,351 | 27,088 |
| 5 | 25,847 | 26,594 | 26,630 |
| 6 | 25,470 | 26,966 | 27,274 |
| 7 | 25,080 | 27,759 | 27,753 |

For the user's 12-miss example, original rank-order balancing gives `[2,2,2,2,1,1,1,1]`; both calibrated policies give `[1,1,1,1,2,2,2,2]`. The model predicts latest-rank H2D completion of 0.536 versus 0.429 ms. At 68 misses, the original vector puts nine copies on GPUs 0–3, giving 2.366 ms predicted completion; both alternatives put nine on GPUs 4–7, giving 2.101 ms, an 11.2% **H2D-service prediction**, not an 11.2% TPOT prediction. At 69–71 misses, their predicted advantage over original Near falls below 0.6%. The PCIe and fast-rank quota rows differ in 31 of 129 miss counts, but their predicted latest-rank time is identical at 65–72 misses, close to this workload's average. Some differing rows are exact predicted ties. This explains why the more elaborate table has little headroom over fast-rank balancing here.

Compiled policy microbenchmarks showed a 4.084 µs lookup-policy median versus 4.056 µs original Near at 12 misses; at 64 and 80 misses the lookup medians were 7.687 and 8.856 µs versus 7.764 and 9.104 µs original Near. Those small differences change sign with input size and are measurement noise at this scale. The lookup adds no meaningful online selection overhead, but its calibrated objective leaves out execution and collective effects. [LOOKUP_OVERHEAD.json](LOOKUP_OVERHEAD.json) preserves all samples and [LOOKUP.json](LOOKUP.json) preserves every quota and source hash.

The live-generation table is useful as a service observation only: output tokens diverged and the optional matched-token control was added after the first live jobs. The matched-token control used the same clean source commit for every policy. No source claims an exact frozen internal routing comparison.
