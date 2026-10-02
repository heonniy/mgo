# E1 actual-trace communication replay

**BLOCKED_PREFLIGHT**. E1 timings are unavailable; E2 was not started.

The first T0 32-KiB preflight exceeded the predeclared 180-second bound and was terminated. All four NCCL communicators logged initialization completion and selected P2P/CUMEM, but the payload exchange never produced a passing receipt. The final logs show shareable-buffer imports and UDS handle mapping. This is not a measured NO_GAP or AMBIGUOUS_GAP result.

Observed worker wait channels included `uvm_gpu_retain_by_uuid` and `uvm_va_space_unregister_gpu`. This locates the observed wait but does not establish its root cause. NCCL 2.28.9 matches the prior successful physical-pilot preflight. No NCCL knob was changed and no driver reset or retry was attempted.

The launcher stopped on timeout, not the memory guard. Peak process-tree RSS was 4.69 GiB; host availability stayed above 1851.92 GiB. Target GPUs 0,1,4,5 returned to zero used memory after cleanup.

## Completed work

- Extracted all 384 original decode dispatch/combine matrices from four SHA256-verified capture files. Count transposes and 4096-byte BF16 rows pass CPU validation.
- Two CPU tests pass: decision boundaries and corrupted receive-count rejection.
- Frozen input and replay protocol were committed before GPU work. The model-free worker, warmup/repetition limits and sentinel validation are implemented; their trace replay has not yet run.
- One T0 preflight attempted and failed. R3 preflight, all four timed cells and every E2 model cell remain unstarted. Zero trace warmups, timed traces or model generations were executed.

## Receipts and boundary

See [failure details and raw-log hashes](trace_comm_preflight_failure.json), [compact NCCL excerpt](trace_comm_T0_preflight_failure.log), [stage JSON](trace_comm_replay.json), [unrun cell CSV](trace_comm_replay.csv), [frozen input](trace_comm_counts.json), and [protocol](TRACE_COMM_REPLAY_PROTOCOL.md).

No R3/T0 ratio exists, so the STRONG_GAP requirement for E2 is unmet. Preserve the failure and stop; do not infer a communication-price contrast or switch servers automatically.
