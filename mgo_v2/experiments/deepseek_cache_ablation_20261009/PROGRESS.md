# DeepSeek ShareGPT cache-capacity ablation

Validated system cells: **0/16**. Input 512, output 64, R4 GPUs 0/1/4/5. Three unfiltered target repeats per completed row; values below are median [minimum, maximum] seconds, with TPOT in seconds per generated token.

| B/rank | Cache | System | TTFT | TPOT | E2E | Status |
|---:|---:|---|---:|---:|---:|---|
| 16 | 20% | main_OURS | — | — | — | PENDING |
| 16 | 20% | MoE-Infinity (repaired) | — | — | — | PENDING |
| 16 | 20% | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| 16 | 20% | llama.cpp balanced | — | — | — | PENDING |
| 16 | 30% | main_OURS | — | — | — | PENDING |
| 16 | 30% | MoE-Infinity (repaired) | — | — | — | PENDING |
| 16 | 30% | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| 16 | 30% | llama.cpp balanced | — | — | — | PENDING |
| 16 | 40% | main_OURS | — | — | — | PENDING |
| 16 | 40% | MoE-Infinity (repaired) | — | — | — | PENDING |
| 16 | 40% | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| 16 | 40% | llama.cpp balanced | — | — | — | PENDING |
| 16 | 50% | main_OURS | — | — | — | PENDING |
| 16 | 50% | MoE-Infinity (repaired) | — | — | — | PENDING |
| 16 | 50% | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| 16 | 50% | llama.cpp balanced | — | — | — | PENDING |

main_OURS uses Near/native experts with prefetch OFF. DeepSeek MoE-Infinity uses EAM eviction priorities with speculative admission OFF. llama.cpp uses whole-layer balanced placement: 1/1/1/1 GPU expert layers at C20/C30, 2/2/2/2 at C40, and 3/3/3/3 at C50. C30 limits expert residency, not total HBM.

Raw frozen token-ID manifests and request lists remain outside Git.
