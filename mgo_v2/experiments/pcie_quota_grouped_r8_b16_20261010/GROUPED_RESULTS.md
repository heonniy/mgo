# R8 B16 grouped new_OURS PCIe quota comparison

Qwen3-30B / ShareGPT / C30 / input512 / output64 / prefetch OFF. All policies use native prefill, compiled dense routing and strict hit-then-miss grouped decode. The same next-token IDs are supplied to each policy; BF16 internal routing can still differ.

| Policy | Repeats | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | First-run actual token differences |
|---|---:|---:|---:|---:|---:|---:|
| Original Near | 3 | 3.157 [1.836, 3.179] | 0.2848 [0.2830, 0.2869] | 21.122 [19.663, 21.232] | 387.840 [385.832, 416.630] | 0 |
| Fast-rank quota | 3 | 3.136 [1.859, 3.150] | 0.2804 [0.2800, 0.2882] | 20.777 [19.527, 21.305] | 394.273 [384.518, 419.523] | 107 |
| PCIe lookup quota | 3 | 3.122 [1.820, 3.137] | 0.2821 [0.2774, 0.2850] | 20.909 [19.293, 21.080] | 391.794 [388.618, 424.599] | 109 |

Each reported value is the mean of two or median of three unfiltered repeats; brackets are the full range. The third repeat rule uses the first-pair TPOT or E2E gap >2%. See [GROUPED_RESULTS.json](GROUPED_RESULTS.json) for all samples, rank fetch counts and raw receipt paths.

The PCIe lookup is 0.59% slower than the fast-rank quota on the reported TPOT. Their full ranges overlap, so this does not establish a lookup gain. The fixed four-seed screen is a separate exploratory follow-up; selection on one timing sample per policy cannot establish a population-best seed.
