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

## Expert subdivision follow-up

Initial and owner-requested restart both reproduced a ~13s increase in the
second generation. Large expert-call increases changed rank (rank3 initially,
rank0 on restart), while other ranks accumulated metadata time. GC differences
were only milliseconds. Neither observation establishes the initiating cause.

`--expert-diagnostics` subdivides decode into ready traversal (including
scheduler readiness/submission), input/weight views, compiled kernel call,
slot-use event, routing-weight lookup and multiplication. Wrap existing CUDA
Event record/query/synchronize and Stream wait/synchronize calls on their own
threads without adding CUDA synchronization. CPU call intervals still cannot
be called GPU kernel durations. The generation parent includes final scheduler
drain to reconcile all recorded main-thread intervals. No primary code change.

CPU regression checks passed for traversal, return positions, BF16 arithmetic,
nested labels and complete hook restoration. Physical diagnostic outputs must
match the uninstrumented full-model warmup with no recompilation.

## Reduce observer effects after expert subdivision

R4_EXPERT_VARIATION completed with validated outputs, but TPOT differed only
~1.1%, so it did not reproduce the large variance. Do not claim stabilization.
The copied diagnostic loop also retains named input/weight temporaries longer
than the original expression; although numerically validated, allocator and
execution-timing effects cannot be excluded. Its many timing ranges perturb
host dispatch. Preserve these limitations when interpreting phase totals.

Next `--kernel-diagnostics` leaves the original expert loop and tensor
lifetimes untouched and only wraps the compiled kernel callable with one host
interval. Retain the original coarse phase/staging/GC accounting. Two LA/B128
runs, same frozen conditions. Use the narrower instrumentation to test whether
large expert-phase variance is inside the compiled callable or outside it.

## Controlled Python thread scheduling intervention

Kernel-only capture did not reproduce the large increase (E2E 90.255/89.119s,
TPOT 1.24021/1.21314s). No stabilization claim follows. Test a specific
remaining hypothesis: Python execution handoff between main dispatch and
expert-staging can affect host dispatch variance. Set Python switch interval
to the original value (recorded, normally 5ms) versus 1ms in ABBA order in
one loaded model. Exactly two generations per setting, with the same coarse
phase instrumentation on all four and no fine kernel wrappers. Preserve
tokens, logical workload, BF16 and frozen routes. This is a diagnostic
intervention, not BR-vs-LA timing or proof that GIL is the sole cause. Record
all samples and context switches. Test B128 first; no unchanged repetition.
If promising, validate B256 and then uninstrumented timing before adoption.
