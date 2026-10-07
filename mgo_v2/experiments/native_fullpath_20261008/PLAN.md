# Native TTFT and TPOT path

The owner requests migration of Python-heavy inference work to C++ where it
improves physical timing, including both prefill and decode. Work only on
GPUs 0, 1, 4, 5. Keep the selected H0 runtime as a measured reference until
the native path passes correctness and timing checks.

This packet extends the existing C++ expert executor into optimized prefill
and replaces the per-expert NumPy MAIN-slot search in both prefill and decode
with a single C++ scan after the controller updates cache roles. It combines
the already validated Numba rank-partial layout with the native executor in
the candidate path. Metadata all-gather, placement decisions, H2D priority,
forward/return collectives, cache roles, BF16 partial sum and barriers retain
the existing semantics. No new persistent HBM allocation is allowed.

Validation sequence:

1. Compile extension and check dynamic slot-role lookup.
2. Four-GPU smoke with optimized prefill, compiled decode layout and native
   executor; require all existing index, state and resource guards to pass.
3. For C30 B16/L256 and B64/L512, capture H0 routing/teacher inputs once.
   Run H0/native/native/H0 with clean cold cache each time, prefetch disabled
   identically, and report TTFT, TPOT, E2E, all samples, token agreement,
   cache/role/controller parity, H2D/peer bytes and peak GPU/host memory.
4. Repeat a bounded B16/L256 comparison with Near policy and P2/T2 prefetch
   enabled in both arms. This covers the selected overlap path that the
   initial BR/prefetch-OFF comparison does not exercise.
5. Only promote a faster validated path. Do not infer that Python itself
   accounts for the whole gain: C++ also changes expert FFN dispatch.

Existing full-pinned memory preflight and 96-GiB host stop remain mandatory.
The supervisor restores model inference loads on owned GPUs after each job.
