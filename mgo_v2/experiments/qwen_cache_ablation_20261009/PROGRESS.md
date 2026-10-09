# Qwen3 ShareGPT main_OURS cache-capacity sweep

Validated cells: **4/4**. R4 GPUs 0/1/4/5, B16/rank, input512/output64. Three unfiltered target repeats; figures are median [minimum, maximum] in seconds, with TPOT in seconds per token.

| Cache | TTFT | TPOT | E2E | H2D GiB, four ranks | Status |
|---:|---:|---:|---:|---:|---|
| 20% | 2.098 [1.974, 3.889] | 0.645 [0.631, 0.649] | 42.721 [41.728, 44.779] | 1913.010 | PASS |
| 30% | 2.053 [1.807, 3.706] | 0.562 [0.556, 0.608] | 37.222 [37.055, 42.010] | 1578.313 | PASS |
| 40% | 2.016 [1.761, 4.236] | 0.538 [0.527, 0.588] | 38.124 [35.193, 38.812] | 1246.421 | PASS |
| 50% | 1.764 [1.763, 3.665] | 0.482 [0.481, 0.482] | 32.147 [32.125, 33.965] | 959.001 | PASS |

main_OURS uses Near/native expert execution, compiled prefill/decode layout, full-pinned host expert source, and prefetch OFF. Raw prompts and token IDs remain outside Git. H2D GiB is the sum of four rank counters per measured batch, including prefill.
