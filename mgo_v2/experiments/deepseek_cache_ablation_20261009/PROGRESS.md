# DeepSeek ShareGPT cache-capacity ablation

Validated system cells: **15/16**. Input 512, output 64, R4 GPUs 0/1/4/5. Three unfiltered target repeats per completed row; values below are median [minimum, maximum] seconds, with TPOT in seconds per generated token.

| B/rank | Cache | System | TTFT | TPOT | E2E | Status |
|---:|---:|---|---:|---:|---:|---|
| 16 | 20% | main_OURS | 0.572 [0.570, 0.895] | 0.288 [0.286, 0.288] | 18.737 [18.708, 18.897] | PASS |
| 16 | 20% | MoE-Infinity (repaired) | 1.619 [1.618, 1.729] | 1.260 [1.254, 1.262] | 81.096 [80.607, 81.149] | PASS |
| 16 | 20% | DeepSpeed ZeRO-Inference | 4.853 [4.833, 6.039] | 4.712 [4.660, 4.716] | 301.984 [298.405, 302.870] | PASS |
| 16 | 20% | llama.cpp balanced | 145.206 [145.197, 145.289] | 0.413 [0.413, 0.414] | 171.265 [171.238, 171.317] | PASS |
| 16 | 30% | main_OURS | 0.603 [0.588, 0.873] | 0.285 [0.283, 0.286] | 18.617 [18.536, 18.706] | PASS |
| 16 | 30% | MoE-Infinity (repaired) | 1.674 [1.660, 1.764] | 1.259 [1.243, 1.259] | 81.010 [79.985, 81.089] | PASS |
| 16 | 30% | DeepSpeed ZeRO-Inference | 4.907 [4.800, 5.948] | 4.638 [4.615, 4.700] | 297.022 [295.678, 302.024] | PASS |
| 16 | 30% | llama.cpp balanced | 145.381 [145.243, 145.453] | 0.413 [0.413, 0.414] | 171.452 [171.293, 171.455] | PASS |
| 16 | 40% | main_OURS | 0.588 [0.578, 0.860] | 0.282 [0.275, 0.286] | 18.604 [17.916, 18.609] | PASS |
| 16 | 40% | MoE-Infinity (repaired) | 1.644 [1.563, 1.772] | 1.217 [1.207, 1.226] | 78.311 [77.821, 78.816] | PASS |
| 16 | 40% | DeepSpeed ZeRO-Inference | 4.795 [4.716, 6.059] | 4.651 [4.616, 4.770] | 297.785 [295.522, 306.568] | PASS |
| 16 | 40% | llama.cpp balanced | 118.479 [118.284, 118.527] | 0.349 [0.348, 0.349] | 140.430 [140.265, 140.486] | PASS |
| 16 | 50% | main_OURS | 0.612 [0.609, 0.862] | 0.282 [0.282, 0.282] | 18.360 [18.358, 18.601] | PASS |
| 16 | 50% | MoE-Infinity (repaired) | 1.624 [1.606, 1.848] | 1.240 [1.238, 1.243] | 79.824 [79.737, 79.920] | PASS |
| 16 | 50% | DeepSpeed ZeRO-Inference | 4.827 [4.806, 5.978] | 4.689 [4.678, 4.828] | 301.390 [299.511, 309.011] | PASS |
| 16 | 50% | llama.cpp balanced | — | — | — | PENDING |

main_OURS uses Near/native experts with prefetch OFF. DeepSeek MoE-Infinity uses EAM eviction priorities with speculative admission OFF. llama.cpp uses whole-layer balanced placement: 1/1/1/1 GPU expert layers at C20/C30, 2/2/2/2 at C40, and 3/3/3/3 at C50. C30 limits expert residency, not total HBM.

Raw frozen token-ID manifests and request lists remain outside Git.
