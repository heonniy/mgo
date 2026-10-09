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

The later C50 diagnostic clean target was 0.682134 s/token versus
0.480953–0.482273 s/token in its primary job. Its instrumented trace showed
rank 1 expert-compute span of 323 ms/token versus about 265–268 ms/token on
peers, with other ranks accumulating ~120-ms/token metadata or return spans.
The trace spans include waiting and do not isolate a pure network cost.

Before the recheck, 16 `hwlee`-owned Nsight `--start-agent` processes from
October 5 were found with PPID 1 and nonexistent `profile-<pid>` session
owners. They consumed CPU time on the shared server. Only those 16 stale
agents were terminated; no model process, active profiler parent, or other
user's job was touched. Their contribution to the timing drift is a
**hypothesis**, to be tested by the post-cleanup C20/C50 jobs.
