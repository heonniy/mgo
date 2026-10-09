# D2 full-resident reference preflight

DeepSeek has 26 routed layers × 64 experts = 1,664 experts. Each BF16 expert
is 16.5 MiB. A deterministic `global_expert_id mod 4` owner map gives exactly
**416 experts/rank**, occupying **6.703 GiB/rank** before the two reserved
physical slots. The C50 matched-route runs used 208 physical slots/rank and
peaked at about 8.9 GiB allocated HBM. Increasing the expert arena from
208 to 418 slots adds about **3.38 GiB/rank**; an estimated 12.3 GiB peak
remains far below the 80-GiB device limit. The worker also checks available
HBM *after* loading dense model weights and requires the full arena plus a
2-GiB margin before allocating it. The pinned host source already exists in
the normal DeepSeek worker at about 26.81 GiB/rank; no second host copy is
created.

The diagnostic preloads every rank-owned expert from pinned CPU memory to
its GPU arena **outside warmup/target timing**, synchronizes the copy, and
marks the static policy owner/slot map resident. It asserts no demand fetches
on every routed layer and zero scheduler H2D bytes in measured generation.
Only GPUs 0/1/4/5 are used, through the existing guarded supervisor.

This is a **C100 static-placement reference**, although it reuses the C50
workload cell for the same prompts and fixed routing trace. Its extra HBM is
diagnostic and exceeds the advertised C20/C50 cache budgets. The static
owner map changes peer traffic and rank workload; its TPOT is not a causal
upper bound on what simply eliminating H2D from the original Near schedules
would achieve. The schedule-preserving no-copy oracle remains separate and
optional under the plan.
