# Main-table repeat stability

Descriptive audit of the three unfiltered target repeats in each completed row. Relative range is (maximum − minimum) / median × 100%. No repeat is excluded or replaced because of this audit. The full samples for every row are in `PROGRESS.json`.

TTFT: median relative range 10.13% across 39 completed rows; maximum 194.21%.
TPOT: median relative range 1.06% across 39 completed rows; maximum 12.05%.
E2E: median relative range 1.23% across 39 completed rows; maximum 21.46%.

Largest relative ranges:

| Dataset | Model | B/rank | Input | System | Metric | Repeats (s) | Relative range |
|---|---|---:|---:|---|---|---|---:|
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | main_OURS | TTFT | 5.222, 1.765, 1.780 | 194.21% |
| ShareGPT | Qwen3 | 16 | 512 | main_OURS | TTFT | 4.284, 1.988, 1.999 | 114.84% |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | main_OURS | TTFT | 5.577, 2.842, 2.811 | 97.34% |
| ShareGPT | Qwen3 | 16 | 1024 | main_OURS | TTFT | 5.221, 2.892, 2.871 | 81.25% |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | DeepSpeed ZeRO-Inference | TTFT | 13.289, 7.533, 7.538 | 76.35% |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | main_OURS | TTFT | 0.887, 0.617, 0.650 | 41.63% |
| ShareGPT | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | TTFT | 7.993, 5.727, 5.691 | 40.20% |
| ShareGPT | Qwen3 | 64 | 512 | main_OURS | TTFT | 6.476, 4.787, 4.674 | 37.64% |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | main_OURS | TTFT | 5.651, 4.185, 4.201 | 34.88% |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | TTFT | 7.488, 5.786, 5.782 | 29.50% |
| ShareGPT | Qwen3 | 16 | 1024 | DeepSpeed ZeRO-Inference | TTFT | 7.097, 5.705, 5.525 | 27.56% |
| ShareGPT | Qwen3 | 64 | 1024 | DeepSpeed ZeRO-Inference | TTFT | 9.247, 7.477, 7.320 | 25.77% |
