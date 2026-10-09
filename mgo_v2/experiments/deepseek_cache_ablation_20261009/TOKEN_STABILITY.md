# Decode token-path stability

All cells use the same 64 frozen target requests and start from an empty dynamic cache after warmup. Agreement is descriptive; it does not change the primary greedy TTFT/TPOT/E2E measurements. Raw request IDs and generated token IDs remain outside Git. Each comparison has 64 requests and 4,096 generated token positions.

## Within a cache setting

| Cache | System | Full-64 request agreement across repeats 1–2 / 1–3 / 2–3 | First-token agreement across repeats 1–2 / 1–3 / 2–3 |
|---:|---|---|---|
| 20% | main_OURS | 64/64 / 64/64 / 64/64 | 64/64 / 64/64 / 64/64 |
| 20% | MoE-Infinity (repaired) | 23/64 / 22/64 / 20/64 | 58/64 / 62/64 / 60/64 |
| 20% | DeepSpeed ZeRO-Inference | 35/64 / 32/64 / 29/64 | 64/64 / 64/64 / 64/64 |
| 20% | llama.cpp balanced | 64/64 / 64/64 / 64/64 | 64/64 / 64/64 / 64/64 |
| 30% | main_OURS | 64/64 / 64/64 / 64/64 | 64/64 / 64/64 / 64/64 |
| 30% | MoE-Infinity (repaired) | 24/64 / 23/64 / 26/64 | 59/64 / 61/64 / 62/64 |
| 30% | DeepSpeed ZeRO-Inference | 31/64 / 34/64 / 32/64 | 64/64 / 64/64 / 64/64 |
| 30% | llama.cpp balanced | 64/64 / 64/64 / 64/64 | 64/64 / 64/64 / 64/64 |
| 40% | main_OURS | 64/64 / 64/64 / 64/64 | 64/64 / 64/64 / 64/64 |
| 40% | MoE-Infinity (repaired) | 23/64 / 26/64 / 29/64 | 60/64 / 61/64 / 61/64 |
| 40% | DeepSpeed ZeRO-Inference | 35/64 / 35/64 / 37/64 | 64/64 / 64/64 / 64/64 |

## C20 reference versus other cache sizes

Compare target repeat 2 at each cache size; first-token agreement, complete 64-token request agreement, and agreement over all token positions are separate.

| System | Comparison | First token | Full 64 tokens | Token positions |
|---|---|---:|---:|---:|
| main_OURS | C20→C30 | 64/64 | 29/64 | 3006/4096 |
| main_OURS | C20→C40 | 64/64 | 32/64 | 3006/4096 |
| MoE-Infinity (repaired) | C20→C30 | 61/64 | 20/64 | 2617/4096 |
| MoE-Infinity (repaired) | C20→C40 | 61/64 | 23/64 | 2590/4096 |
| DeepSpeed ZeRO-Inference | C20→C30 | 64/64 | 36/64 | 3177/4096 |
| DeepSpeed ZeRO-Inference | C20→C40 | 64/64 | 31/64 | 2962/4096 |
| llama.cpp balanced | C20→C30 | 64/64 | 64/64 | 4096/4096 |

Capacity changes can alter BF16 execution order and, after a token diverges, later routing demand. Therefore the measured end-to-end results describe actual greedy runs at each capacity; traffic differences do not isolate capacity under a bit-identical future decode trajectory.
