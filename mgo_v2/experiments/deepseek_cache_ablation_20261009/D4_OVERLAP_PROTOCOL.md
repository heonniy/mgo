# D4 focused overlap ablation

After D1–D3, test the specific ready-first/H2D-overlap mechanism with the
same fixed continuation and frozen-router trace. For each capacity C20 and
C50, run one guarded job on GPUs 0/1/4/5 with a separate normal warmup and
four cold-cache targets in order **normal, serial, serial, normal**. Each arm
therefore gets two clean uninstrumented targets; the balanced order limits
simple chronological drift. Do not add a fifth target if the two per-arm
measurements agree within the prior timing tolerance.

The serial arm waits on all **local current-layer demand H2D copies after
dispatch and before native expert execution**. The normal arm retains
ready-first execution while the dedicated copy stream runs. Both arms use
the same Near policy, expert budget, C++ per-expert executor, pinned host
source, prefetch OFF and identical route/next-token inputs. No global rank
barrier or change in collective packet format is introduced.

Require exact request IDs, router-file hash, fed-token hash, actual H2D
bytes, expert-group counts, cache capacity and finite outputs across arms.
Record predicted-token agreement separately; BF16 execution order may
change. This intervention also collapses ready waves, so a TPOT difference
is the **net** effect of allowing H2D/compute overlap versus altered wave
grouping. It is not a pure PCIe latency or standalone GEMM measurement.

Keep all raw target/token records outside Git. Commit two-repeat means,
full ranges, per-rank H2D/wave/host-wait counters, exact-parity validation
and the source revision. Restore the owner model loads at job end.
