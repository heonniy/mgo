# Main-table memory audit

C30 limits resident expert weights, not total HBM. MoE-Infinity expert peak charges and llama.cpp resident expert weights are checked against the per-GPU budget for every completed row. The llama.cpp per-GPU value is derived from its audited equal-size expert layers and exact layer ownership. The HBM column is the highest 1 Hz supervisor sample across physical GPUs 0/1/4/5 during each job; it may miss shorter peaks. PyTorch allocated/reserved peaks come from worker counters where available. The difference between those counters does not isolate KV, attention, allocator, and native workspace costs. Pinned host memory is shown only when the worker measured it.

| Dataset | Model | B/rank | Input | System | C30 expert budget/GPU (GiB) | llama expert resident/GPU (GiB) | Sampled peak HBM/GPU (GiB) | PyTorch allocated peak/GPU (GiB) | PyTorch reserved peak/GPU (GiB) | Pinned host/rank (GiB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ShareGPT | Qwen3 | 16 | 512 | main_OURS | 4.04–4.05 | not measured | 13.97–14.22 | 8.56–8.65 | 10.96–10.97 | 54.00–54.00 |
| ShareGPT | Qwen3 | 16 | 512 | DeepSpeed ZeRO-Inference | 4.04–4.05 | not measured | 12.02–12.49 | 7.05–7.05 | 9.53–9.53 | 14.22–14.22 |
| ShareGPT | Qwen3 | 16 | 512 | MoE-Infinity (repaired) | 4.04–4.05 | not measured | 12.00–13.79 | 2.63–2.69 | 5.79–6.96 | not measured |
| ShareGPT | Qwen3 | 16 | 512 | llama.cpp balanced | 4.04–4.05 | 3.38–3.38 | 5.76–6.39 | not measured | not measured | not measured |
| ShareGPT | Qwen3 | 64 | 512 | main_OURS | 4.04–4.05 | not measured | 20.03–21.04 | 13.53–13.68 | 16.98–17.85 | 54.00–54.00 |
| ShareGPT | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | 4.04–4.05 | not measured | 16.09–16.56 | 9.59–9.59 | 13.60–13.60 | 14.22–14.22 |
| ShareGPT | Qwen3 | 64 | 512 | MoE-Infinity (repaired) | 4.04–4.05 | not measured | 30.34–33.59 | 10.42–10.68 | 24.13–26.76 | not measured |
| ShareGPT | Qwen3 | 64 | 512 | llama.cpp balanced | 4.04–4.05 | 3.38–3.38 | 9.14–9.52 | not measured | not measured | not measured |
| ShareGPT | Qwen3 | 16 | 1024 | main_OURS | 4.04–4.05 | not measured | 14.96–15.67 | 10.21–10.34 | 11.95–12.47 | 54.00–54.00 |
| ShareGPT | Qwen3 | 16 | 1024 | DeepSpeed ZeRO-Inference | 4.04–4.05 | not measured | 13.40–13.87 | 7.80–7.80 | 10.90–10.90 | 14.22–14.22 |
| ShareGPT | Qwen3 | 16 | 1024 | MoE-Infinity (repaired) | 4.04–4.05 | not measured | 18.20–20.34 | 5.23–5.35 | 11.98–13.73 | not measured |
| ShareGPT | Qwen3 | 16 | 1024 | llama.cpp balanced | 4.04–4.05 | 3.38–3.38 | 6.51–7.08 | not measured | not measured | not measured |
| ShareGPT | Qwen3 | 64 | 1024 | main_OURS | 4.04–4.05 | not measured | 30.87–39.03 | 19.95–20.27 | 27.62–35.84 | 54.00–54.00 |
| ShareGPT | Qwen3 | 64 | 1024 | DeepSpeed ZeRO-Inference | 4.04–4.05 | not measured | 21.34–21.81 | 13.07–13.07 | 18.85–18.85 | 14.22–14.22 |
| ShareGPT | Qwen3 | 64 | 1024 | MoE-Infinity (repaired) | 4.04–4.05 | not measured | 50.00–57.86 | 20.81–21.32 | 43.79–51.02 | not measured |
| ShareGPT | Qwen3 | 64 | 1024 | llama.cpp balanced | 4.04–4.05 | 3.38–3.38 | 12.14–12.77 | not measured | not measured | not measured |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | main_OURS | 2.00–2.01 | not measured | 11.04–11.42 | 7.47–7.49 | not measured | 26.81–26.81 |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | DeepSpeed ZeRO-Inference | 2.00–2.01 | not measured | 9.32–9.79 | 5.19–5.20 | 6.84–6.85 | 7.31–7.31 |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | MoE-Infinity (repaired) | 2.00–2.01 | not measured | 10.96–12.20 | 4.33–4.52 | 7.04–7.74 | not measured |
| ShareGPT | DeepSeekV2Lite | 16 | 512 | llama.cpp balanced | 2.00–2.01 | 1.03–1.03 | 2.58–3.05 | not measured | not measured | not measured |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | main_OURS | 2.00–2.01 | not measured | 21.63–22.15 | 16.14–16.24 | not measured | 26.81–26.81 |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | DeepSpeed ZeRO-Inference | 2.00–2.01 | not measured | 20.88–21.35 | 15.02–15.10 | 18.41–18.41 | 7.31–7.31 |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | MoE-Infinity (repaired) | 2.00–2.01 | not measured | 32.48–34.24 | 17.21–18.00 | 28.62–29.90 | not measured |
| ShareGPT | DeepSeekV2Lite | 64 | 512 | llama.cpp balanced | 2.00–2.01 | 1.03–1.03 | 3.69–4.00 | not measured | not measured | not measured |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | main_OURS | 2.00–2.01 | not measured | 14.53–14.77 | 10.35–10.39 | not measured | 26.81–26.81 |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | DeepSpeed ZeRO-Inference | 2.00–2.01 | not measured | 12.58–13.17 | 7.27–7.29 | 10.10–10.23 | 7.31–7.31 |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | MoE-Infinity (repaired) | 2.00–2.01 | not measured | 26.10–26.87 | 13.68–14.08 | 21.92–22.64 | not measured |
| ShareGPT | DeepSeekV2Lite | 16 | 1024 | llama.cpp balanced | 2.00–2.01 | 1.03–1.03 | 2.83–3.26 | not measured | not measured | not measured |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | main_OURS | 2.00–2.01 | not measured | 35.50–38.11 | 27.79–27.95 | not measured | 26.81–26.81 |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | DeepSpeed ZeRO-Inference | 2.00–2.01 | not measured | 35.82–36.29 | 28.40–28.50 | 33.35–33.35 | 7.31–7.31 |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | MoE-Infinity (repaired) | 2.00–2.01 | not measured | 63.92–64.88 | 16.14–19.41 | 59.66–60.63 | not measured |
| ShareGPT | DeepSeekV2Lite | 64 | 1024 | llama.cpp balanced | 2.00–2.01 | 1.03–1.03 | 4.67–4.85 | not measured | not measured | not measured |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | main_OURS | 4.04–4.05 | not measured | 13.96–14.22 | 8.60–8.66 | 10.95–10.97 | 54.00–54.00 |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | DeepSpeed ZeRO-Inference | 4.04–4.05 | not measured | 12.02–12.49 | 7.05–7.05 | 9.53–9.53 | 14.22–14.22 |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | MoE-Infinity (repaired) | 4.04–4.05 | not measured | 12.09–13.33 | 2.63–2.69 | 5.87–6.49 | not measured |
| LMSYS-Chat-1M | Qwen3 | 16 | 512 | llama.cpp balanced | 4.04–4.05 | 3.38–3.38 | 5.76–6.39 | not measured | not measured | not measured |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | main_OURS | 4.04–4.05 | not measured | 20.04–20.25 | 13.48–13.73 | 16.89–17.03 | 54.00–54.00 |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | DeepSpeed ZeRO-Inference | 4.04–4.05 | not measured | 16.09–16.56 | 9.59–9.59 | 13.60–13.60 | 14.22–14.22 |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | MoE-Infinity (repaired) | 4.04–4.05 | not measured | 28.83–35.75 | 10.42–10.68 | 22.62–28.95 | not measured |
| LMSYS-Chat-1M | Qwen3 | 64 | 512 | llama.cpp balanced | 4.04–4.05 | 3.38–3.38 | 9.14–9.52 | not measured | not measured | not measured |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | main_OURS | 4.04–4.05 | not measured | 14.96–15.22 | 10.24–10.35 | 11.95–11.97 | 54.00–54.00 |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | DeepSpeed ZeRO-Inference | 4.04–4.05 | not measured | 13.40–13.87 | 7.80–7.80 | 10.90–10.90 | 14.22–14.22 |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | MoE-Infinity (repaired) | 4.04–4.05 | not measured | 17.45–19.26 | 5.23–5.35 | 11.24–13.02 | not measured |
| LMSYS-Chat-1M | Qwen3 | 16 | 1024 | llama.cpp balanced | 4.04–4.05 | 3.38–3.38 | 6.51–7.08 | not measured | not measured | not measured |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | main_OURS | 4.04–4.05 | not measured | 32.36–40.99 | 19.95–20.35 | 29.12–37.79 | 54.00–54.00 |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | DeepSpeed ZeRO-Inference | 4.04–4.05 | not measured | 21.34–21.81 | 13.06–13.07 | 18.85–18.85 | 14.22–14.22 |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | MoE-Infinity (repaired) | 4.04–4.05 | not measured | 52.21–54.87 | 20.81–21.32 | 46.00–48.65 | not measured |
| LMSYS-Chat-1M | Qwen3 | 64 | 1024 | llama.cpp balanced | 4.04–4.05 | 3.38–3.38 | 12.14–12.77 | not measured | not measured | not measured |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 512 | main_OURS | 2.00–2.01 | not measured | 11.15–11.41 | 7.47–7.48 | not measured | 26.81–26.81 |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 512 | DeepSpeed ZeRO-Inference | 2.00–2.01 | not measured | 9.32–9.79 | 5.17–5.18 | 6.84–6.85 | 7.31–7.31 |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 512 | MoE-Infinity (repaired) | 2.00–2.01 | not measured | 10.86–12.27 | 4.33–4.52 | 6.90–7.81 | not measured |
| LMSYS-Chat-1M | DeepSeekV2Lite | 16 | 512 | llama.cpp balanced | 2.00–2.01 | 1.03–1.03 | 2.58–3.05 | not measured | not measured | not measured |
| LMSYS-Chat-1M | DeepSeekV2Lite | 64 | 512 | main_OURS | 2.00–2.01 | not measured | 21.03–21.43 | 16.17–16.23 | not measured | 26.81–26.81 |
