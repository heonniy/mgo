# Root-cause priority: host phase variation

Owner explicitly requires diagnosing and stabilizing timing variation before
resuming performance-candidate search. Complete the active bounded async
metadata study, then run a separate two-generation same-policy diagnostic.
Use LA/B128 first, R4 GPUs0,1,4,5, BF16, frozen decode64, V3/P2/T2, torch
staging, unique combine and async metadata. No new route capture.

Instrumentation is diagnostic-only, without Nsight/CUPTI or timed per-copy
CUDA events. Record per-thread wall and thread CPU intervals for existing
runtime phases and pageable-to-pinned staging. Record Python GC pauses.
Nested phases have inclusive and exclusive totals; never sum different
threads or add GC to phases as independent wall time. Capture the whole
generation so uninstrumented main-thread work is visible as exclusive time.
Copy steps are the main event observed on entry, not causal copy provenance.

Two diagnostic repetitions only. Validate tokens and frozen workload for each,
and require identical controller/scheduler counts across repetitions. Compare
phase deltas between faster/slower generations; a phase correlation is not a
root cause. If variance does not reproduce, say so rather than selecting or
adding favorable repetitions. CPU time may include spin waits; non-CPU wall
may include GIL, I/O, CUDA waits or descheduling. Attribution must respect
these limits and lead to a controlled test of the strongest supported cause.

These are not extra primary BR-vs-LA samples and cannot support a gain claim.
Primary repeat gates remain unchanged. Keep scientific/foreign-process/OOM
safety checks and restore owned model-load jobs when no experiment is active.
