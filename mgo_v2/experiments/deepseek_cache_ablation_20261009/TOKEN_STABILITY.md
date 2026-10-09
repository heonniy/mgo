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

## C20 reference versus other cache sizes

Compare target repeat 2 at each cache size; first-token agreement, complete 64-token request agreement, and agreement over all token positions are separate.

| System | Comparison | First token | Full 64 tokens | Token positions |
|---|---|---:|---:|---:|
| main_OURS | C20→C30 | 64/64 | 29/64 | 3006/4096 |
| MoE-Infinity (repaired) | C20→C30 | 61/64 | 20/64 | 2617/4096 |

Capacity changes can alter BF16 execution order and, after a token diverges, later routing demand. Therefore the measured end-to-end results describe actual greedy runs at each capacity; traffic differences do not isolate capacity under a bit-identical future decode trajectory.
