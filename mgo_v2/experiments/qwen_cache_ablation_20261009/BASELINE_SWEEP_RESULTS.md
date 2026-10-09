# Qwen cache-capacity baseline sweep

ShareGPT, R4 GPUs 0/1/4/5, local B16/input512/output64. C30 reuses three validated same-request, same-budget targets. New cells take two clean targets, with one bounded third if TPOT or E2E differs by over 2% but no more than 5%; the already-started C20 MoE-Infinity job has three. All samples and full ranges are retained.

| Cache | System | TTFT median [range], s | TPOT median [range], s/token | E2E median [range], s | Receipt |
|---:|---|---:|---:|---:|---|
| C20 | infinity | 5.207 [5.177, 5.247] | 3.143 [3.138, 3.144] | 203.185 [202.946, 203.291] | `qca_baseline_c20_infinity_r3_v1 (3 repeats; THREE)` |
| C20 | deepspeed | 5.886 [5.402, 6.371] | 4.045 [4.043, 4.047] | 260.741 [260.137, 261.346] | `qca_baseline_c20_deepspeed_r2_v1 (2 repeats; TWO_STABLE)` |
| C20 | llama | 161.936 [161.788, 162.083] | 0.525 [0.525, 0.525] | 195.028 [194.887, 195.169] | `qca_baseline_c20_llama_r2_v1 (2 repeats; TWO_STABLE)` |
| C30 | infinity | 5.192 [5.182, 5.219] | 3.227 [3.213, 3.264] | 208.505 [207.659, 210.809] | `mt2_qwen_sharegpt_b16_l512_infinity_r3_v3 (C30 reuse) (3 repeats; THREE)` |
| C30 | deepspeed | 5.571 [5.451, 6.312] | 4.122 [4.091, 4.135] | 265.995 [263.176, 266.055] | `mt2_qwen_sharegpt_b16_l512_deepspeed_r3_v1 (C30 reuse) (3 repeats; THREE)` |
| C30 | llama | 145.070 [145.007, 145.185] | 0.481 [0.480, 0.481] | 175.331 [175.287, 175.486] | `mt2_qwen_sharegpt_b16_l512_llama_r3_v1 (C30 reuse) (3 repeats; THREE)` |
| C40 | infinity | 5.548 [5.241, 5.577] | 3.204 [3.189, 3.269] | 207.416 [206.176, 211.501] | `qca_baseline_c40_infinity_r2_v1 + qca_baseline_c40_infinity_followup_r1_v2 (3 repeats; THIRD)` |
| C40 | deepspeed | 5.951 [5.469, 6.433] | 4.112 [4.106, 4.117] | 264.985 [264.157, 265.814] | `qca_baseline_c40_deepspeed_r2_v1 (2 repeats; TWO_STABLE)` |
| C40 | llama | 129.525 [129.470, 129.580] | 0.436 [0.435, 0.436] | 156.963 [156.912, 157.014] | `qca_baseline_c40_llama_r2_v1 (2 repeats; TWO_STABLE)` |
| C50 | infinity | pending | pending | pending | — |
| C50 | deepspeed | pending | pending | pending | — |
| C50 | llama | pending | pending | pending | — |

Expert residency/placement or live-parameter budget checks are recorded per cell in `BASELINE_SWEEP_RESULTS.json`. llama.cpp statically holds balanced full expert layers, so its quantized GPU residency is not a dynamic cache hit-rate measurement. DeepSpeed limits all live parameters, including non-experts, under its cap. Raw requests, tokens, logs and GPU resource samples remain outside Git.
