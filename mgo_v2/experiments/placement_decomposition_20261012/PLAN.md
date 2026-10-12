# Stage-2 placement cost decomposition (serial + inline H2D, FAST quota): load imbalance vs comm

One untimed diagnostic job per arm and model. Arms: FAST+Near (`NEAR_FAST`),
FAST+Random (`FAST_RANDOM`), FAST+Worst (`FAST_WORST`).
- Qwen3 R8: `run_qwen_r8_job.py --ours-mode A --h2d-serial-ablation --inline-demand-h2d
  --post-generation-diagnostic` (live tokens, instrumented pass after the timed target).
- DeepSeek R8: `run_deepseek_r8_job.py ... --teacher-tokens`, with `MGO_DEEPSEEK_CACHE_DIAG=1`
  (the target repeat itself is instrumented). The worker records per-layer dispatch, serial H2D wait,
  expert compute and return spans.

Per decode layer, from rank spans:
- load-imbalance cost = max_r(compute) - mean_r(compute), summed over layers;
- H2D-wait imbalance = max_r(wait) - mean_r(wait), as a check (same quota, so it should match across arms);
- comm = dispatch + return spans and bytes. These spans include waiting for the slowest peer, so the
  comm share is also reported as mean dispatch+return time net of that wait.
