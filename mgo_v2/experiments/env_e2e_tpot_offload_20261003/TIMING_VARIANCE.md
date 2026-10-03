# Timing variability checkpoint — 2026-10-04

P/CA-rep/Env1 repeats 1 and 2 took 608.151 and 470.431 seconds E2E.
Decode wall was 605.387 versus 465.999 seconds: the difference is in decode,
not model loading or compilation. All eight output-token/final-cache hashes
match; worker source, frozen action schedule and no-compilation gate agree.
Both phases sampled a maximum GPU temperature of 41 C. No resource guard
or foreign-GPU-process failure occurred. This does not exclude CPU contention,
frequency changes, NUMA/memory effects or short unsampled contention.

The external safety monitor is itself a confound candidate: median resource
collection took 24.979 versus 16.763 seconds per scan. It traverses process
memory mappings to obtain PSS and then waits only five seconds before the next
scan. The monitor is outside the worker timer but runs concurrently with it.
The correlation does not prove monitor-induced slowdown; both may respond to
another host-side cause. No CPU scheduling, H2D, kernel or NCCL decomposition
was captured, so the present records cannot attribute the 137.721-second gap.
The whole-decode CUDA interval also includes waits and is not isolated expert
compute time.

Retain both samples and full ranges. Do not interpret one-repeat policy or
transport ordering as stable. Continue only the frozen repetition/noise gate;
no profiler, new diagnostic GPU run, or mid-series runtime/monitor change was
made for this checkpoint. If permitted repeats remain noisy, report that
limitation rather than selecting the fastest sample or declaring a mechanism.
See timing_variance_diagnostic.json for source receipts and exact figures.
