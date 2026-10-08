# Two-model C30 main-table progress

Validated rows: **10/64**.

Each completed row has three unfiltered clean measurements. Times are seconds; TPOT is seconds per generated token and includes attention.

| Dataset | Model | B/rank | Input | System | TTFT median [range] | TPOT median [range] | E2E median [range] | Status |
|---|---|---:|---:|---|---:|---:|---:|---|
| ShareGPT | Qwen3 | 16 | 512 | main_OURS | 1.999 [1.988, 4.284] | 0.637 [0.634, 0.641] | 42.342 [41.956, 44.406] | PASS |
| ShareGPT | Qwen3 | 16 | 512 | DeepSpeed ZeRO-Inference | 5.571 [5.451, 6.312] | 4.122 [4.091, 4.135] | 265.995 [263.176, 266.055] | PASS |
| ShareGPT | Qwen3 | 16 | 512 | MoE-Infinity (repaired) | 5.859 [5.568, 6.089] | 3.208 [3.175, 3.208] | 207.693 [206.096, 207.973] | PASS |
| ShareGPT | Qwen3 | 16 | 512 | llama.cpp balanced | 145.070 [145.007, 145.185] | 0.481 [0.480, 0.481] | 175.331 [175.287, 175.486] | PASS |
| ShareGPT | Qwen3 | 64 | 512 | main_OURS | 4.787 [4.674, 6.476] | 0.741 [0.739, 0.764] | 51.362 [51.314, 54.589] | PASS |
| ShareGPT | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | 5.727 [5.691, 7.993] | 4.748 [4.733, 4.749] | 304.907 [303.925, 307.117] | PASS |
| ShareGPT | Qwen3 | 64 | 512 | MoE-Infinity (repaired) | — | — | — | PENDING |
| ShareGPT | Qwen3 | 64 | 512 | llama.cpp balanced | — | — | — | PENDING |
| ShareGPT | Qwen3 | 16 | 1024 | main_OURS | — | — | — | PENDING |
| ShareGPT | Qwen3 | 16 | 1024 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| ShareGPT | Qwen3 | 16 | 1024 | MoE-Infinity (repaired) | — | — | — | PENDING |
| ShareGPT | Qwen3 | 16 | 1024 | llama.cpp balanced | — | — | — | PENDING |
| ShareGPT | Qwen3 | 64 | 1024 | main_OURS | — | — | — | PENDING |
| ShareGPT | Qwen3 | 64 | 1024 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| ShareGPT | Qwen3 | 64 | 1024 | MoE-Infinity (repaired) | — | — | — | PENDING |
| ShareGPT | Qwen3 | 64 | 1024 | llama.cpp balanced | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | main_OURS | 0.650 [0.617, 0.887] | 0.287 [0.285, 0.294] | 18.833 [18.747, 19.112] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | DeepSpeed ZeRO-Inference | 4.905 [4.885, 5.944] | 4.745 [4.697, 4.745] | 303.821 [300.817, 304.848] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | MoE-Infinity (repaired) | 1.673 [1.643, 1.749] | 1.247 [1.241, 1.248] | 80.292 [79.848, 80.295] | PASS |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | llama.cpp balanced | 145.843 [145.516, 146.156] | 0.414 [0.413, 0.414] | 171.831 [171.575, 172.217] | PASS |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | main_OURS | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | MoE-Infinity (repaired) | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | llama.cpp balanced | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | main_OURS | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | MoE-Infinity (repaired) | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | llama.cpp balanced | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | main_OURS | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | MoE-Infinity (repaired) | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | llama.cpp balanced | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | main_OURS | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | MoE-Infinity (repaired) | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | llama.cpp balanced | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | main_OURS | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | MoE-Infinity (repaired) | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | llama.cpp balanced | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | main_OURS | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | MoE-Infinity (repaired) | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | llama.cpp balanced | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | main_OURS | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | MoE-Infinity (repaired) | — | — | — | PENDING |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | llama.cpp balanced | — | — | — | PENDING |
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
