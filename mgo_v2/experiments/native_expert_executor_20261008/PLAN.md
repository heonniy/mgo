# C++ expert execution migration

Owner requests C++ execution using origin/main EP coslot as reference.
Reference main commit: 27edf65. Inspect EP Python executor and archer-coslot
C++ dispatcher, especially cache slot validation and last-compute events.
The old dispatcher adds slot-to-parameter D2D copies and host stream sync;
do not import those into the H0/full-pinned overlap runtime.

Implement an opt-in C++ loop per currently ready wave. Per-expert GEMMs
remain separate (not grouped GEMM). Read cache slots directly, release GIL,
use current compute stream, retain weighted outputs in original group order.
Keep controller, routing, full-pinned H2D scheduler, T2, and return unchanged.
No added all-rank barrier, wait-to-fill threshold or persistent graph cache.
Python handles one readiness snapshot and completion event per wave; the
per-expert gather/FFN/weight loop moves to C++. Existing H0 remains available.

Validation: build with one compiler job; GPU0/1/4/5 only under existing
supervisor and OOM guards. Small BF16 numerical, ordering, empty and
not-ready dependency tests; bounded all-ready synthetic executor timing.
Then B64/R4/C60/input256 baseline vs native, same warmup/reset and 64 decode
steps initially, prefetch OFF, identical3678 MAIN +8 reserved and T2 overlap.
Use fixed teacher inputs/routes for executor isolation if feasible; report
native greedy divergence separately. No unbounded repetitions or other policy
matrix. Performance claims require clean timing, not diagnostic CPU spans.

Numerical implementation differs in kernel packaging from torch.compile;
report BF16 error and token agreement instead of claiming bitwise parity.
Do not promote the new default before physical validation.
