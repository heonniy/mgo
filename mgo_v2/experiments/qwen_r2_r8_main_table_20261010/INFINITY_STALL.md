# R8 MoE-Infinity target stall and bounded repair

The first R8/ShareGPT/B16/input512/output64/C30 full run
(`infinity_full_v1`) completed warmup but stopped making progress during
target repeat 1. All 133 worker threads were sleeping and all eight GPUs
reported 0% utilization over repeated samples. The owned guard terminated
that attempt and restored only its managed loads. A second trace-enabled
64-token run (`infinity_full_v2`) completed warmup and exactly four target
tokens, then made no progress for over 200 seconds. Its in-process Python
watchdog could not run during the native wait; the owner stopped it through
the guard. Both attempts are preserved outside Git under
`/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs/` and are
excluded from the main table.

A bounded full-batch, input512, eight-output-token diagnostic
(`infinity_full_v3`) passed. Its warmup admitted zero EAM speculative
candidates across 384 routed-layer calls; its target admitted **591,448**
across the same 384 calls. That large target-only submission volume is a
plausible cause of the later native wait, but the short target passed, so
the evidence does not prove causality. Do not report its timing as a
64-token baseline.

The spec-off R8 smoke (`infinity_smoke_v2`) passed with zero admitted
speculative candidates in both warmup and target, and its peak EAM
accounting stayed inside C30. The next comparison keeps the exact C30 expert budget and EAM eviction
priority updates, while disabling speculative transfer admission for both
R2 and R8. A low-cost per-token progress receipt and external 120-second
watchdog now capture a native stall independently of the Python worker's
GIL. The first spec-off job is a functional smoke; only subsequent
untraced 64-token full runs are headline eligible.
