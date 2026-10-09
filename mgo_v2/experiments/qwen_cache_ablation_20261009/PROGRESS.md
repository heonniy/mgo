# Qwen3 ShareGPT main_OURS cache-capacity sweep

Validated cells: **2/4**. R4 GPUs 0/1/4/5, B16/rank, input512/output64. Three unfiltered target repeats; figures are median [minimum, maximum] in seconds, with TPOT in seconds per token.

| Cache | TTFT | TPOT | E2E | H2D GiB, four ranks | Status |
|---:|---:|---:|---:|---:|---|
| 20% | 2.098 [1.974, 3.889] | 0.645 [0.631, 0.649] | 42.721 [41.728, 44.779] | 1913.010 | PASS |
| 30% | 2.053 [1.807, 3.706] | 0.562 [0.556, 0.608] | 37.222 [37.055, 42.010] | 1578.313 | PASS |
| 40% | — | — | — | — | PENDING |
| 50% | — | — | — | — | PENDING |

main_OURS uses Near/native expert execution, compiled prefill/decode layout, full-pinned host expert source, and prefetch OFF. Raw prompts and token IDs remain outside Git. H2D GiB is the sum of four rank counters per measured batch, including prefill.
