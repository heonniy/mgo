# Two-model C30 main-table progress

Validated rows: **47/64**.

Each completed row has three unfiltered clean measurements. Times are seconds; TPOT is seconds per generated token and includes attention. See [MEMORY_AUDIT.md](MEMORY_AUDIT.md) for the expert budget and total HBM measurements.

| Dataset | Model | B/rank | Input | System | TTFT median [range] | TPOT median [range] | E2E median [range] | Status |
|---|---|---:|---:|---|---:|---:|---:|---|
| ShareGPT | Qwen3 | 16 | 512 | main_OURS | 1.999 [1.988, 4.284] | 0.637 [0.634, 0.641] | 42.342 [41.956, 44.406] | PASS |
| ShareGPT | Qwen3 | 16 | 512 | DeepSpeed ZeRO-Inference | 5.571 [5.451, 6.312] | 4.122 [4.091, 4.135] | 265.995 [263.176, 266.055] | PASS |
| ShareGPT | Qwen3 | 16 | 512 | MoE-Infinity (repaired) | 5.192 [5.182, 5.219] | 3.227 [3.213, 3.264] | 208.505 [207.659, 210.809] | PASS |
| ShareGPT | Qwen3 | 16 | 512 | llama.cpp balanced | 145.070 [145.007, 145.185] | 0.481 [0.480, 0.481] | 175.331 [175.287, 175.486] | PASS |
| ShareGPT | Qwen3 | 64 | 512 | main_OURS | 4.787 [4.674, 6.476] | 0.741 [0.739, 0.764] | 51.362 [51.314, 54.589] | PASS |
| ShareGPT | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | 5.727 [5.691, 7.993] | 4.748 [4.733, 4.749] | 304.907 [303.925, 307.117] | PASS |
| ShareGPT | Qwen3 | 64 | 512 | MoE-Infinity (repaired) | 8.071 [8.022, 8.189] | 3.756 [3.726, 3.774] | 244.811 [242.821, 245.763] | PASS |
| ShareGPT | Qwen3 | 64 | 512 | llama.cpp balanced | 578.346 [577.541, 578.413] | 1.399 [1.399, 1.402] | 666.507 [665.670, 666.732] | PASS |
| ShareGPT | Qwen3 | 16 | 1024 | main_OURS | 2.892 [2.871, 5.221] | 0.631 [0.629, 0.636] | 42.939 [42.615, 44.866] | PASS |
| ShareGPT | Qwen3 | 16 | 1024 | DeepSpeed ZeRO-Inference | 5.705 [5.525, 7.097] | 4.117 [4.053, 4.133] | 265.061 [260.854, 267.483] | PASS |
| ShareGPT | Qwen3 | 16 | 1024 | MoE-Infinity (repaired) | 6.274 [6.144, 6.333] | 3.210 [3.210, 3.214] | 208.488 [208.388, 208.818] | PASS |
| ShareGPT | Qwen3 | 16 | 1024 | llama.cpp balanced | 287.679 [287.678, 287.945] | 0.480 [0.479, 0.480] | 317.896 [317.862, 318.184] | PASS |
| ShareGPT | Qwen3 | 64 | 1024 | main_OURS | 7.895 [7.821, 8.728] | 0.590 [0.583, 0.591] | 44.961 [44.610, 45.957] | PASS |
| ShareGPT | Qwen3 | 64 | 1024 | DeepSpeed ZeRO-Inference | 7.477 [7.320, 9.247] | 4.685 [4.659, 4.693] | 302.962 [301.024, 304.398] | PASS |
| ShareGPT | Qwen3 | 64 | 1024 | MoE-Infinity (repaired) | 13.494 [13.304, 13.935] | 3.769 [3.744, 3.786] | 250.773 [249.350, 252.440] | PASS |
| ShareGPT | Qwen3 | 64 | 1024 | llama.cpp balanced | 1149.674 [1148.766, 1150.098] | 1.405 [1.402, 1.405] | 1238.015 [1237.255, 1238.605] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | main_OURS | 0.650 [0.617, 0.887] | 0.287 [0.285, 0.294] | 18.833 [18.747, 19.112] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | DeepSpeed ZeRO-Inference | 4.905 [4.885, 5.944] | 4.745 [4.697, 4.745] | 303.821 [300.817, 304.848] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | MoE-Infinity (repaired) | 1.673 [1.643, 1.749] | 1.247 [1.241, 1.248] | 80.292 [79.848, 80.295] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | llama.cpp balanced | 145.843 [145.516, 146.156] | 0.414 [0.413, 0.414] | 171.831 [171.575, 172.217] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | main_OURS | 1.984 [1.967, 2.061] | 0.302 [0.299, 0.308] | 21.038 [20.803, 21.481] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | DeepSpeed ZeRO-Inference | 4.870 [4.828, 5.658] | 4.748 [4.722, 4.820] | 304.001 [302.296, 309.324] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | MoE-Infinity (repaired) | 3.717 [3.696, 3.759] | 1.269 [1.252, 1.275] | 83.661 [82.591, 84.060] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | llama.cpp balanced | 580.575 [580.263, 580.659] | 1.345 [1.337, 1.347] | 665.024 [664.866, 665.419] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | main_OURS | 0.960 [0.917, 1.125] | 0.285 [0.284, 0.288] | 19.033 [18.889, 19.090] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | DeepSpeed ZeRO-Inference | 4.787 [4.774, 5.990] | 4.667 [4.657, 4.694] | 300.003 [298.193, 300.482] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | MoE-Infinity (repaired) | 2.396 [2.383, 2.474] | 1.222 [1.220, 1.233] | 79.455 [79.221, 80.065] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | llama.cpp balanced | 289.553 [288.516, 289.755] | 0.414 [0.413, 0.415] | 315.679 [314.584, 315.799] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | main_OURS | 4.075 [3.834, 4.257] | 0.318 [0.316, 0.318] | 23.996 [23.841, 24.317] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | DeepSpeed ZeRO-Inference | 5.276 [5.260, 6.271] | 4.737 [4.733, 4.746] | 304.234 [303.732, 304.446] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | MoE-Infinity (repaired) | 6.717 [6.511, 6.812] | 1.375 [1.361, 1.392] | 93.335 [92.276, 94.507] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | llama.cpp balanced | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | main_OURS | 1.780 [1.765, 5.222] | 0.517 [0.515, 0.578] | 34.335 [34.252, 41.618] | PASS |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | DeepSpeed ZeRO-Inference | 5.650 [5.583, 6.464] | 4.271 [4.230, 4.276] | 274.701 [272.928, 274.951] | PASS |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | MoE-Infinity (repaired) | 5.220 [5.072, 5.601] | 3.302 [3.291, 3.313] | 213.605 [212.581, 213.803] | PASS |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | llama.cpp balanced | 146.072 [146.030, 146.291] | 0.486 [0.486, 0.487] | 176.722 [176.643, 176.898] | PASS |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | main_OURS | 4.201 [4.185, 5.651] | 0.607 [0.606, 0.622] | 42.408 [42.377, 44.858] | PASS |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | 5.786 [5.782, 7.488] | 4.915 [4.913, 4.985] | 315.400 [315.323, 321.517] | PASS |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | MoE-Infinity (repaired) | 8.085 [7.996, 8.297] | 3.941 [3.934, 3.950] | 256.592 [255.952, 256.838] | PASS |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | llama.cpp balanced | 581.718 [581.422, 582.466] | 1.408 [1.400, 1.408] | 670.438 [669.613, 671.157] | PASS |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | main_OURS | 2.842 [2.811, 5.577] | 0.636 [0.633, 0.684] | 42.872 [42.725, 48.640] | PASS |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | DeepSpeed ZeRO-Inference | 5.638 [5.620, 6.853] | 4.245 [4.208, 4.255] | 273.666 [270.758, 274.299] | PASS |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | MoE-Infinity (repaired) | 5.953 [5.789, 5.998] | 3.127 [3.064, 3.166] | 202.997 [198.990, 205.236] | PASS |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | llama.cpp balanced | 288.799 [288.747, 288.889] | 0.485 [0.485, 0.486] | 319.332 [319.275, 319.478] | PASS |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | main_OURS | 7.691 [7.653, 8.557] | 0.603 [0.603, 0.608] | 45.702 [45.646, 46.838] | PASS |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | DeepSpeed ZeRO-Inference | 7.538 [7.533, 13.289] | 4.897 [4.882, 5.189] | 316.073 [315.080, 340.220] | PASS |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | MoE-Infinity (repaired) | 13.334 [13.185, 13.571] | 3.761 [3.750, 3.799] | 250.119 [249.821, 252.662] | PASS |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | llama.cpp balanced | 1162.287 [1161.994, 1162.588] | 1.410 [1.407, 1.413] | 1250.932 [1250.828, 1251.579] | PASS |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 512 | main_OURS | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 512 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 512 | MoE-Infinity (repaired) | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 512 | llama.cpp balanced | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 512 | main_OURS | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 512 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 512 | MoE-Infinity (repaired) | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 512 | llama.cpp balanced | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 1024 | main_OURS | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 1024 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 1024 | MoE-Infinity (repaired) | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 1024 | llama.cpp balanced | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 1024 | main_OURS | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 1024 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 1024 | MoE-Infinity (repaired) | — | — | — | PENDING |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 1024 | llama.cpp balanced | — | — | — | PENDING |

DeepSeek MoE-Infinity uses EAM eviction priorities with speculative prefetch disabled after a native expert-wait stall; Qwen MoE-Infinity retains speculative EAM prefetch. See `DEEPSEEK_INFINITY_ADAPTATION.md`.

DeepSeek MoE-Infinity uses eager attention for the three cells that fit and SDPA for B64/L1024, where eager attention OOMs. The selected backend is recorded per row in `PROGRESS.json`; see `INFINITY_DEEPSEEK_SDPA_REPAIR.md`.
