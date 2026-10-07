# Owner-requested four-repeat OURS rerun

Owner asked for4–5 repeats of TTFT/TPOT/E2E at B16/L256 and B64/L512. Fixed at exactly4 per cell, no automatic fifth repeat or outlier filtering. This explicit request supersedes the earlier no-extra-repetition limit.

R4/C30, physical GPUs0/1/4/5 only, Near/H0/full-pinned, prefill-optimized + prefill-layout-fast. Runtime arithmetic/placement/cache/decode unchanged from3f0a501. Same frozen ShareGPT-long manifests; global batch64/256; input256/512; output64.

Per-cell preparation remains unchanged: one-token metadata/index validation using disjoint warmup inputs, then full64-token disjoint warmup. Four unprofiled measured batches follow, resetting dynamic expert/cache/history before each. No separate post-run profiler or extra diagnostic passes. Keep every raw repeat, report mean/sample-SD/range and spread, and compare tokens/state against previous repaired-runtime outputs.

Run B16 first, then B64. Exclusive supervised execution,384GiB host-start/96GiB abort guards. Restore owned idle model loads on0/1/4/5 on exit; never touch2/3/6/7.
