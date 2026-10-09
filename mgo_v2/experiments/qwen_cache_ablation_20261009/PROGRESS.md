# Qwen3 ShareGPT main_OURS cache-capacity sweep

Validated cells: **0/4**. R4 GPUs 0/1/4/5, B16/rank, input512/output64. Three unfiltered target repeats; figures are median [minimum, maximum] in seconds, with TPOT in seconds per token.

| Cache | TTFT | TPOT | E2E | H2D GiB, four ranks | Status |
|---:|---:|---:|---:|---:|---|
| 20% | — | — | — | — | PENDING |
| 30% | — | — | — | — | PENDING |
| 40% | — | — | — | — | PENDING |
| 50% | — | — | — | — | PENDING |

main_OURS uses Near/native expert execution, compiled prefill/decode layout, full-pinned host expert source, and prefetch OFF. Raw prompts and token IDs remain outside Git. H2D GiB is the sum of four rank counters per measured batch, including prefill.
