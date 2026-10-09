# Qwen C20–C50 four-system comparison

ShareGPT R4/local B16/input512/output64. All values below are median [full range] over clean targets. main_OURS uses its later two-repeat post-cleanup timing recheck; baseline C30 uses validated earlier three-repeat measurements and other baseline cells use fresh guarded runs. This is an observational cross-system comparison across different execution dates, not a paired frozen-route intervention.

| Cache | System | TTFT, s | TPOT, s/token | E2E, s | Targets |
|---:|---|---:|---:|---:|---:|
| C20 | main_OURS (post-cleanup) | 2.683 [1.752, 3.614] | 0.503 [0.502, 0.504] | 34.392 [33.515, 35.270] | 2 |
| C20 | infinity | 5.207 [5.177, 5.247] | 3.143 [3.138, 3.144] | 203.185 [202.946, 203.291] | 3 |
| C20 | deepspeed | 5.886 [5.402, 6.371] | 4.045 [4.043, 4.047] | 260.741 [260.137, 261.346] | 2 |
| C20 | llama | 161.936 [161.788, 162.083] | 0.525 [0.525, 0.525] | 195.028 [194.887, 195.169] | 2 |
| C30 | main_OURS (post-cleanup) | 2.700 [1.767, 3.632] | 0.495 [0.494, 0.496] | 33.870 [33.017, 34.723] | 2 |
| C30 | infinity | 5.192 [5.182, 5.219] | 3.227 [3.213, 3.264] | 208.505 [207.659, 210.809] | 3 |
| C30 | deepspeed | 5.571 [5.451, 6.312] | 4.122 [4.091, 4.135] | 265.995 [263.176, 266.055] | 3 |
| C30 | llama | 145.070 [145.007, 145.185] | 0.481 [0.480, 0.481] | 175.331 [175.287, 175.486] | 3 |
| C40 | main_OURS (post-cleanup) | 2.687 [1.762, 3.611] | 0.493 [0.493, 0.493] | 33.745 [32.835, 34.655] | 2 |
| C40 | infinity | 5.548 [5.241, 5.577] | 3.204 [3.189, 3.269] | 207.416 [206.176, 211.501] | 3 |
| C40 | deepspeed | pending | pending | pending | — |
| C40 | llama | pending | pending | pending | — |
| C50 | main_OURS (post-cleanup) | 2.730 [1.763, 3.697] | 0.489 [0.488, 0.490] | 33.532 [32.620, 34.444] | 2 |
| C50 | infinity | pending | pending | pending | — |
| C50 | deepspeed | pending | pending | pending | — |
| C50 | llama | pending | pending | pending | — |

main_OURS keeps native C++ expert execution, compiled prefill/decode index paths, Near placement, pinned CPU expert source and prefetch OFF. The three baselines retain their audited main-table implementations and their own memory semantics. Full raw per-target statistics, cache budgets and job provenance are in `BASELINE_SWEEP_RESULTS.json` and `TIMING_RECHECK.json`.
