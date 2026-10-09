# Adaptive timing recheck

The clean C20 target in the diagnostic job was 0.507364 s/token, versus
0.631015–0.649052 s/token in the three-repeat primary C20 job. The same
request IDs, full output tokens, H2D bytes, and final cache hashes match on
all four ranks. The diagnostic hooks are installed **after** that clean
target, so the difference exposes cross-job timing drift rather than a
different measured request or cache state.

Preserve the original unfiltered primary rows. Use the adjacent C20
diagnostic clean target, C50 diagnostic clean target, then one short clean
C20 recheck as an A–B–A bracket. The recheck uses the identical frozen
manifest, main_OURS flags, cold cache after warmup, and two unfiltered target
repeats. Validate all tokens and H2D bytes against the original C20 run.
If the bracket does not stabilize C20, report the cross-job range and avoid
claiming a precise C20→C50 TPOT gain. Neither diagnostic replay may replace
clean timing. This recheck adds no policy/runtime changes.
