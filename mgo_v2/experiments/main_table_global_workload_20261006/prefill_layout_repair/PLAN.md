# Owner-requested CPU layout and device-index repair

Keep the prefill-optimized Near/H0/full-pinned path, BF16 operation order, global routing/policy and cache budgets unchanged. GPUs0/1/4/5 only.

Replace per-layer global route sorting/Python-list construction with a Numba linear packet builder in exact peer/token/expert order. Keep NumPy arrays through packing; transfer only forward/group/target indices consumed by rank-partial return. Do not build unused expert-return order/combine arrays or recopy selected IDs. No grouped GEMM, extra overlap or policy change.

Validation: independent legacy CPU parity across empty/uneven/variable-length/single-owner and both full prefill shapes (28 cases). Physical all48-layer exact index comparisons on warmup inputs plus existing metadata/numerical guards. Two full64-token primaries per cell with cold reset after warmup. B64 first, then B16. Compare all generated tokens/cache hashes to previous optimized primaries. One separate one-token diagnostic at the end of each loaded model verifies targeted component costs; never use instrumented timing as headline. No automatic extra repeats or outlier removal.

Keep legacy implementation available as a reference; the repair is enabled with prefill-optimized + prefill-layout-fast flags. Preserve raw failures/results and restore owned model loads on0/1/4/5.
