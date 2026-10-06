# R4 BR/Near rerun on current H0 full-pinned runtime

Owner requests eight combinations: local B16/B64 x input256/512 x BR/LA_CA_NEAR.
Use GPUs 0,1,4,5, C30, existing strict experiment selected frozen inputs,
existing sample/DP/placement seeds, and decode32 (33 outputs including prefill).
No new trace or seed search. CPU P2 reference replay is regenerated because
this is the current H0/full-pinned V3 overlap runtime, with P2/T2 prefetch,
unique combine, async metadata and BF16; old strict layer barriers are absent.
Near uses its existing demand placement and LA predictor-based prefetch placement.

One unmeasured correctness/warmup per policy and one primary measurement per
requested combination: eight primary runs, eight warmups. Alternate policy
measurement order across cells. Keep all samples; single-shot results cannot
establish repeat stability. Record wall TTFT/TPOT/E2E, full rank receipts,
CPU state/copy bounds, finite logits, per-policy warm token parity, and any
cross-policy BF16 token differences. No detailed diagnostics or extra policies.
One model and rank-private pinned pool are reused across cells; caches reset
per policy run. Pinned allocation is outside timing. H0 only: no graph cache.

Require 384 GiB host available at launch; abort below96 GiB; one-hour bound.
Stop/restore only owned model loads on GPUs0,1,4,5; never touch2,3,6,7.
Commit each completed cell and final result. Preserve previous raw results.
