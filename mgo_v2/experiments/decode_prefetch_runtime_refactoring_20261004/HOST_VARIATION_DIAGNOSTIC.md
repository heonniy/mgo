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

### Switch-interval result and CPU placement control

GIL ABBA: original TPOT 1.21065/1.42800s (16.47% difference); 1ms
TPOT 1.26073/1.36980s (8.29%). All four got progressively slower, so do not
attribute the smaller middle-pair spread to the interval. Reject 1ms as an
established stabilization. Work and tokens remained validated. In the middle
pair rank1 expert host CPU increased ~12.97s, while ranks0/3 metadata CPU
increased ~7.88/~10.18s; this again suggests one rank's delay propagating,
without distinguishing dispatch/spin/memory/scheduling as its cause.

Next diagnostic ABBA changes only process-local CPU placement: existing
24-CPU rank mask versus main pinned to its first CPU, staging to its second,
and existing helper threads to the remaining 22. No host affinity or foreign
process changes, no new buffers. Restore the original rank mask for the last
generation. Keep default Python switch interval. Two samples each, B128 first.
This tests migration/shared-core contention within a rank, not physical NUMA
placement (the guest does not expose it). Record actual masks in every sample.

Affinity setup audit caught that runtime reset replaces the staging thread.
The first affinity launch was terminated during warmup before any measurements;
it is an infrastructure FAIL, not a timing sample. Apply masks after reset
to the live thread and record its actual affinity. Relaunch as V2.

V2 was also terminated during its first isolated sample: a native affinity
audit found two threads on each staging singleton CPU. The copy operation
creates an OpenMP helper lazily, inheriting the staging thread's mask; assigning
all existing helpers elsewhere does not cover future helpers. This is a
diagnostic configuration error, not evidence about the original runtime's
jitter. Preserve V2's completed baseline and affinity_midrun_audit.json.
V3 gives staging plus its lazy copy helper two dedicated CPUs (indices1:3),
main one CPU (index0), existing helpers the remaining21 CPUs. Main/staging
masks remain disjoint. Validate live masks during the isolated diagnostic.

### V3 outcome and clean validation

All four generations passed frozen workload and output validation. Original
TPOT 1.22180/1.42387s (15.28%); isolated TPOT 1.43599/1.39407s (2.96%).
Actual native masks confirm two staging threads share two dedicated CPUs.
This is only a candidate for stabilization: the original measurements bracket
a longer time interval, and time/order effects remain confounded with masks.
Do not claim causality or acceleration (the isolated samples were slower).

Next use the existing uninstrumented BR/LA paired harness on B128/B256, with
identical isolated masks for both policies. Apply masks after every runtime
reset, including warmups. Two stable pairs stop; at most one third as already
authorized. No phase hooks and no process/GPU scans during MEASURE. Record
actual thread masks and include the helper source in the common fingerprint.
No default change: isolated_cpu_threads is explicit in both case records.

## Clean isolated-mask pilot failed; fix copy-team placement

B128 clean paired timing remained unstable: BR TPOT 1.38437/1.38633/1.41659;
LA 1.36697/1.45338/1.46785. B256 was cancelled during warmup, with no timed
samples, after this failed pilot. Do not adopt the shared two-CPU staging mask.

The LA second sample changed rank2 staging endpoint CPU97->98 and staging
CPU time ~32.15->46.14s; other staging ranks changed <1s. BR third changed
rank1 staging endpoint CPU25->26 and CPU time ~31.5->35.7s. Endpoint CPU
is not a migration trace; these observations motivate, but do not prove,
copy-team interference as the initiating cause. Counters/output were valid.

Add opt-in fixed_staging_team: initialize the worker's two-thread OpenMP
copy pool with a private stage zero-fill outside timing, require exactly one
new native helper, and pin staging and helper to distinct singleton CPUs.
The scheduler records IDs/masks and waits for setup success before accepting
work. Repeat this on each runtime reset. No extra expert copy/GPU operation
or buffer; default behavior unchanged. Fail closed if helper identity cannot
be established. This Linux/Torch-two-thread experiment is not a portable
default. Test B128 first, and only proceed to B256 if B128 is stable.

Validation: CPU successive-worker initialization passed; GPU BF16 9-MiB
copy contents, byte/count accounting, post-copy singleton masks, and worker
recreation passed. Existing delayed-CUDA scheduler-lock regression also passed.

## Fixed copy-team clean validation: stable on both batches

FIXED_TEAM_RESULTS.json and COPY_TEAM_STABILIZATION.json record completed
R4/0145 B128/B256 frozen decode64 BF16 BR/LA validation. Both batches stopped
at two counterbalanced pairs: every policy's E2E and TPOT relative difference
was below 0.5%. No third or extra unchanged repetitions were run. Logical
controller counters, cache hashes and output hashes matched between repeats.
Staging endpoint CPUs matched fixed singleton masks. Physical scheduler
accounting has two documented differences: B128 BR rank0 reclassified two
background copies as urgent; B256 BR rank0 issued one extra 9-MiB prefetch
copy instead of cancelling it (32268+1 cancelled vs32269+0). Existing frozen
proof copies+cancelled remains satisfied. Do not claim identical physical
copy counts or hide these scheduling differences.

B128 means: BR E2E98.9328s / TPOT1.38752s; LA E2E97.6809s / TPOT1.37027s.
B256 means: BR E2E124.9668s / TPOT1.67165s; LA E2E123.2640s / TPOT1.64945s.
Descriptive paired TPOT gains are ~1.24% and ~1.33%; both CI95s include zero.
Do not inflate these into established positive improvements. Respect the
owner's two-stable-pairs stopping rule rather than repeating for significance.

The result supports copy-team placement/scheduling and initialization as a
practical stabilization on this host. It does not prove that endpoint CPU
changes alone caused all historical jitter. Keep fixed_staging_team explicit
for follow-up timing; default runtime selection and overall optimization goal
remain open. GPU model load workers are restored after the bounded run.
