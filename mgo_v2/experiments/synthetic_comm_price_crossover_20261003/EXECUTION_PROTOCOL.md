# Frozen execution conventions

## C0/C1

CPU stages use hidden CUDA, BLAS/OMP threads=1, one process and a hard 4-GiB
address-space bound. Verify the original frontier outputs against their
committed hashes and every raw capture size/SHA256. Verify all historical
producer source blobs at the original source commit recorded in provenance.
C1 imports the exact historical ReplicaReplay blob by its recorded SHA256;
it does not use newer callback extensions. Apply every frozen event through
the existing independent action/state applier and check original full,
prefill, decode totals and final state hashes. Preserve original greedy
replication during prefill as well as decode.

C0 solves all pairwise positive rational intersections and determines the
actual lower envelope; it does not assume all adjacent rho points survive.
Exact numerators/denominators and ties are published. Commit C0 before GPU work.

## C2 timing and gate

Only GPU 0/1/4/5, stable T0 with NCCL_CUMEM_ENABLE=0 and no R3 run. One bounded
32-KiB T0 transport smoke verifies P2P/IPC before timing; INFO is absent in
timing workers. Pause/restore only our active model workers on those four GPUs.
Never touch GPU 2/3/6/7 or restart our stopped workers there.

Ten worker invocations, one per rho/pass, each contain two timing cells:
actual and matched self-only. Pass0 rho ascending, actual then self; pass1 rho
descending, self then actual. This yields exactly twenty cells, each one
untimed warmup and five timed full 384-event traces. No extra repetitions.
Each condition uses the same 768 collective calls and its original diagonal
self-row counts. Self-only clears every off-diagonal count without turning
remote routes into self traffic. Validate every payload outside timing.
Preallocate buffers and all CUDA events outside timing. Retain whole-trace
max-rank CUDA and wall times, plus event/sum aggregations as secondary.

For each cell, take per-repeat max across ranks, then median over five traces.
T_remote = median_actual - median_self, without clipping negative values.
The owner explicitly selected normalization by rho0 T_remote (not per byte).
Within each pass, compare rho 0/.125/.25/.5 in adjacent order:
normalized_next <= 1.10 * normalized_previous, where
normalized_rho = T_remote(rho)/T_remote(0).
Require every such T_remote >=0 and positive rho0 premium for a defined
normalization. Separately require abs(actual-self) <= 0.10*self at rho .75.
Failure in either pass means TIMING_UNSTABLE; no new samples or NCCL tuning.
Publish all individual gate checks. Do not reinterpret a failed time model
as evidence against the exact C0 resource-price shift.

## Conditional C3/C4

If C2 fails, do not publish a time-price model or crossover. If C2 passes,
reuse only the exact existing concurrent-four H2D median/p90 calibration;
no new H2D timing. Sum event max-rank fetch counts times one-copy cost.
For C4, each policy's self/remote coefficients are the medians of its two
pass-specific medians/premiums. With two passes this equals their arithmetic
mean. Use rational representations of recorded decimal coefficients when
solving the algebraic envelope; this does not imply exact hardware timing.
Publish median/p90 H2D model sweeps, all positive envelope crossovers and ties.
Gamma is a synthetic remote-price multiplier, never a bandwidth ratio.

Keep R3 only as previously documented unstable context. Stop after the fixed
matrix and result commit; no new model, H2D, R3, batch, cache or replica policy.

For the zero-peer rho=.75 control, preserve raw actual-minus-self differences
and apply the original ±10% C2 agreement gate. The proposed C4 default is
T_remote(.75)=0 after that gate passes, because gamma must price only remote
traffic. This convention was raised with the owner before timing; absent an
alternate preference it prevents amplifying a measured zero-traffic residual.
