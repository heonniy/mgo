# Checkpoint A: physical critical-path microbench

R4 physical GPUs 0/1/4/5, Env1 verified P2P/IPC; BF16. No full-model inference, new trace capture, profiler, or Stage B/oracle run.

## A1: matched-volume split shape

| Packets | Bytes/packet | Shape | Median max-local completion ms | / balanced |
|---:|---:|---|---:|---:|
| 1389 | 4120 | BALANCED | 0.031472 | 1.000 |
| 1389 | 4120 | PAIR_HOT | 0.030528 | 0.970 |
| 1389 | 4120 | DEST_HOT | 0.051456 | 1.635 |
| 1389 | 4120 | SRC_HOT | 0.044992 | 1.430 |
| 1389 | 4120 | TRACE_BR | 0.031520 | 1.002 |
| 1389 | 4120 | TRACE_FCA | 0.031616 | 1.005 |
| 1389 | 4096 | BALANCED | 0.031136 | 1.000 |
| 1389 | 4096 | PAIR_HOT | 0.030352 | 0.975 |
| 1389 | 4096 | DEST_HOT | 0.041920 | 1.346 |
| 1389 | 4096 | SRC_HOT | 0.038704 | 1.243 |
| 1389 | 4096 | TRACE_BR | 0.032352 | 1.039 |
| 1389 | 4096 | TRACE_FCA | 0.032000 | 1.028 |
| 1432 | 4120 | BALANCED | 0.032144 | 1.000 |
| 1432 | 4120 | PAIR_HOT | 0.031744 | 0.988 |
| 1432 | 4120 | DEST_HOT | 0.053488 | 1.664 |
| 1432 | 4120 | SRC_HOT | 0.046688 | 1.452 |
| 1432 | 4120 | TRACE_BR | 0.031856 | 0.991 |
| 1432 | 4120 | TRACE_FCA | 0.032304 | 1.005 |
| 1432 | 4096 | BALANCED | 0.030848 | 1.000 |
| 1432 | 4096 | PAIR_HOT | 0.031008 | 1.005 |
| 1432 | 4096 | DEST_HOT | 0.042432 | 1.376 |
| 1432 | 4096 | SRC_HOT | 0.041424 | 1.343 |
| 1432 | 4096 | TRACE_BR | 0.032128 | 1.041 |
| 1432 | 4096 | TRACE_FCA | 0.032592 | 1.057 |
| 1528 | 4120 | BALANCED | 0.032336 | 1.000 |
| 1528 | 4120 | PAIR_HOT | 0.031328 | 0.969 |
| 1528 | 4120 | DEST_HOT | 0.054224 | 1.677 |
| 1528 | 4120 | SRC_HOT | 0.048720 | 1.507 |
| 1528 | 4120 | TRACE_BR | 0.032832 | 1.015 |
| 1528 | 4120 | TRACE_FCA | 0.033680 | 1.042 |
| 1528 | 4096 | BALANCED | 0.032304 | 1.000 |
| 1528 | 4096 | PAIR_HOT | 0.031296 | 0.969 |
| 1528 | 4096 | DEST_HOT | 0.043504 | 1.347 |
| 1528 | 4096 | SRC_HOT | 0.043280 | 1.340 |
| 1528 | 4096 | TRACE_BR | 0.032352 | 1.001 |
| 1528 | 4096 | TRACE_FCA | 0.032000 | 0.991 |

Feature correlations are descriptive across the bounded matrix; see JSON. Raw unscaled trace matrices are also measured and retained separately.

## A2: arrival skew

Measured-delay vs incremental-tail slope: 0.999; Spearman: 0.992. GPU-delay targets 0/0.10/0.25/0.50/1/2 ms, two matrices, all four delayed ranks. NCCL residency includes arrival waiting.

## A3: compiled expert service

2/4/8-expert bundle median absolute relative prediction error: 10.12%. Unstable rank/row calibrations after bounded blocks: 1.

The exact runtime expert function is compiled with dynamic=True/fullgraph=True and real Qwen3 checkpoint weights. Power ladder 1–512; extra observed bundle row sizes avoid interpolation. All samples, drift, and any unstable cells are retained.

## Decision and limits

This is Checkpoint A, not a policy-gain result. Stage B has not started. An oracle remains gated on the separate held-out model-validation requirements.

## Expert power ladder

| Rows | Median across rank medians (ms) | P10 pooled (ms) | P90 pooled (ms) |
|---:|---:|---:|---:|
| 1 | 0.020048 | 0.019776 | 0.020512 |
| 2 | 0.020208 | 0.019424 | 0.020544 |
| 4 | 0.020232 | 0.019808 | 0.020544 |
| 8 | 0.020176 | 0.020000 | 0.020640 |
| 16 | 0.020240 | 0.020032 | 0.020672 |
| 32 | 0.020456 | 0.020256 | 0.020800 |
| 64 | 0.020816 | 0.020416 | 0.021184 |
| 128 | 0.021216 | 0.020893 | 0.021536 |
| 256 | 0.022512 | 0.022304 | 0.022912 |
| 512 | 0.026848 | 0.026080 | 0.033312 |

All primary power-ladder cells meet the 2% final-block drift gate. Supplemental rank1/n18 remains unstable (20.25% final-block drift); retain all three blocks and do not treat that calibration as stable. Row-count/service-time Spearman by rank: {'0': 0.9515151515151514, '1': 0.9057792600832962, '2': 0.9878787878787878, '3': 0.9787279253249042}.

CUDA graph GPU execution span of the exact compiled expert function, including inter-kernel gaps; not a CUPTI sum of kernel durations.

Matched synthetic source/destination-hot shapes are 1.24–1.68x balanced. At matched volumes, the selected BR/FCA-shaped matrices are within approximately 6% of balanced; the shape-only experiment does not explain the earlier multi-fold return-NCCL residency gap. The approximately one-for-one skew/tail relationship establishes sensitivity to arrival skew in isolation, not that expert skew alone caused the earlier TPOT loss.

Checkpoint stop instruction: AGENT_TASK.md explicitly says “STOP and publish MICROBENCH_RESULTS before doing Stage B.” No new online policy or oracle was implemented.

- No cross-GPU absolute event clock: max rank duration is common-rendezvous approximation.
- CUDA graph GPU service excludes Python/controller/H2D effects; not a full-runtime predictive validation.
- No causal effect on prior FCA TPOT established by isolated microbench alone.
- All valid observations retained; bounded max-three tau blocks can remain unstable.
