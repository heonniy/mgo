# Decode layout compiled path

Owner authorizes implementing the reviewed decode layout optimization.
Stage1 only: preserve Near placement, metadata/history, prefetch, cache semantics,
BF16 accumulation and communication. Reuse exact canonical compiled rank-partial
packet builder in decode. Remove unused expert-order return metadata and repeated
selected D2H. Build layer expert->physical slot lookup once after plan_current;
never cache across promotion/eviction. GPU metadata/controller migration deferred.

Opt-in --decode-layout-fast, fused BF16 rank-partial only. Existing default stays
available. CPU differential test covers100 cases including empty/uneven decode
B16/B32/B64 and physical-role swaps/eviction. GPU warmup compares all3024 decode
layer indices/slots per rank against old builder; not primary timing.

One cell R4/C30/B32/input512/output64, GPUs0/1/4/5, Near/H0/full-pinned, prior
prefill optimizations ON. New baseline2 repeats then candidate2, disjoint full
warmup each. Expert/cache/history reset before primaries. No profiler and no
extra repeats. All samples retained; compare mean TPOT/E2E plus TTFT, ranges.
Exact generated tokens, final cache/roles/controller and H2D byte/copy counters
compared to fresh baseline. Preserve mismatches; do not call changed work a gain.
CPU compiled work must be warmed before primary timing. Host384/96GiB guards.
Do not claim a GPU controller or remove metadata D2H dependency in this stage.
