# Bounded grouped R8 ShareGPT sample-seed screen

Qwen3-30B, R8/C30/local-B16/input512/output64, strict hit-then-miss `new_OURS`, compiled dense, prefetch OFF. The four seeds were frozen before timing; each screen cell has one unfiltered target, so the screen is exploratory. The same continuation token IDs are supplied to fast-rank and PCIe policies, while their BF16 routing and predicted tokens can differ from the Near teacher; counts are in the JSON.

| Seed | Near TPOT | Fast-rank TPOT | PCIe-lookup TPOT | PCIe gain vs fast |
|---:|---:|---:|---:|---:|
| 11 | 0.2882 | 0.2865 | 0.2843 | +0.75% |
| 23 | 0.2850 | 0.2845 | 0.2798 | +1.65% |
| 37 | 0.2836 | 0.2787 | 0.2818 | -1.10% |
| 53 | 0.2870 | 0.2796 | 0.2762 | +1.19% |

Seed 23 had the largest **observed single-shot** PCIe gain. Fresh confirmations on that seed:

| Policy | Repeats | TPOT center (s/token) | Full TPOT range |
|---|---:|---:|---:|
| Fast-rank quota | 3 | 0.2807 | [0.2788, 0.2848] |
| PCIe lookup quota | 3 | 0.2794 | [0.2787, 0.2850] |

Confirmed PCIe gain versus fast-rank quota: **+0.46%** on the selected seed. The full TPOT ranges overlap, and the original fixed workload showed a 0.60% regression, so a reliable lookup advantage is not established. Keep the current main_OURS Near default. The first Fast/PCIe confirmation predictions differed at 128 of 8,192 output-token positions despite the same supplied continuation IDs. The selected seed is not a claim of a population optimum. All raw attempts and metrics are retained in [SEED_RESULTS.json](SEED_RESULTS.json).
