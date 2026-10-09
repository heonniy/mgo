# Qwen baseline attempt failures

`qca_baseline_c40_infinity_followup_r1_v1` failed with worker exit -11 during
warmup prefill, after the model had loaded and before any target was measured.
The Python fault trace points through MoE-Infinity's Qwen expert predictor and
expert-tracer update. This is a native segmentation fault; the Python frame
does not identify the crashing native instruction or prove the root cause.
The guarded receipt shows at least 1.79 TiB available host memory and at
least 76 GiB free on each owner GPU during the failed interval, so there is
no evidence of an OOM. The supervisor restored the four owner inference loads
and left no GPU occupants. The earlier `qca_baseline_c40_infinity_r2_v1`
target repeats both passed; their 3.269/3.189 s/token difference triggered
this single third-target follow-up. Preserve the failed attempt and retry the
one-target follow-up under a new label. If the fault repeats, inspect the
MoE-Infinity native predictor/tracer path before running more cells.
