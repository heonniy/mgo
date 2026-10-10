# R8 serial GEMM + inline H2D: NEAR vs FAST

Serial ablation (`--h2d-serial-ablation`, `--ours-mode A`): native per-expert executor; the
host waits for all of a layer's demand copies before any expert compute (no H2D/compute
overlap). Both arms use `--inline-demand-h2d`, prefetch OFF, and the frozen Near teacher tokens.
Same Qwen3 ShareGPT R8/C30/local-B16/input512/output64 cell.

- NEAR (`LA_CA_NEAR`): baseline.
- FAST (`NEAR_FAST`): remainder to the fast GPUs 4-7, Near placement.

Three rounds with alternating order (N,F / F,N / N,F), 2 target repeats per job (6 samples per arm).
