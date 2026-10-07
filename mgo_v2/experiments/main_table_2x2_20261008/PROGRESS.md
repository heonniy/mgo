# Two-model C30 main-table progress

Validated rows: **3/64**.

Each completed row has three unfiltered clean measurements. Times are seconds; TPOT is seconds per generated token and includes attention.

| Dataset | Model | B/rank | Input | System | TTFT median [range] | TPOT median [range] | E2E median [range] | Status |
|---|---|---:|---:|---|---:|---:|---:|---|
| ShareGPT | Qwen3 | 16 | 512 | main_OURS | 1.999 [1.988, 4.284] | 0.637 [0.634, 0.641] | 42.342 [41.956, 44.406] | PASS |
| ShareGPT | Qwen3 | 16 | 512 | DeepSpeed ZeRO-Inference | 5.571 [5.451, 6.312] | 4.122 [4.091, 4.135] | 265.995 [263.176, 266.055] | PASS |
| ShareGPT | Qwen3 | 16 | 512 | MoE-Infinity (repaired) | — | — | — | PENDING |
| ShareGPT | Qwen3 | 16 | 512 | llama.cpp balanced | — | — | — | PENDING |
| ShareGPT | Qwen3 | 64 | 512 | main_OURS | — | — | — | PENDING |
| ShareGPT | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
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
| ShareGPT | DeepSeekV2Lite | 16 | 512 | DeepSpeed ZeRO-Inference | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | MoE-Infinity (repaired) | — | — | — | PENDING |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | llama.cpp balanced | — | — | — | PENDING |
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
