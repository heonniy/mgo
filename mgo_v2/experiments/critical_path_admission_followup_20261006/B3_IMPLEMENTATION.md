# B3 implementation checkpoint

Plan: f60c84b. Scope: C30/B128, R4 physical 0/1/4/5, BR/FCA,
BF16 V3 P2/T2, frozen64 input. C60 remains gated.

H0 remains the original compiled expert call. H1 is opt-in and keyed by
(slot, exact rows). Graphs refer directly to the live slot weights. Each entry
has persistent BF16 input/output; kernel temporaries share a graph pool and
never survive a call. The graph copies its result to its own output before
return. One current stream orders scratch writes, replay, and the unchanged
routing multiplication. No graph-private weight copy or change in combine
order, readiness selection, placement, H2D scheduler or collectives.

Before H1 profiling: full64 H0 discovery; drain and stop H2D staging; graph
construction; full64 H1 validation with zero new entries/compilations. Check
expert outputs exactly on first decode step; require identical full64 tokens,
CPU cache/role proofs, controller counters, copy counts/bytes and packet bytes.
A missing signature always invalidates execution. Construction checks predicted
scratch against 12 GiB reserved free memory and stops below 10 GiB free.

Separate B2-style diagnostics use the fixed first8 decode steps. Preparation
and graph capture precede profiler start. H1 compiled-kernel host ranges include
input scratch copy plus replay; GPU ranges include graph kernels/output copy.
These diagnostic wall times are not primary TPOT.

First diagnostic order: H1/BR, H0/FCA, H1/FCA, H0/BR. This resolves H1
feasibility before new baseline captures. Clean counterbalanced64 timing is
subsequent, only after correctness and diagnostic gates. No oracle, grouped
GEMM, policy retuning, or automatic C60 expansion.

Shared-server safeguards retain host memory and selected-GPU checks. Foreign
processes on 2/3/6/7 are recorded and untouched. Owned idle workers resume only
on 0/1/4/5 after experiment exit. All valid observations and failures persist.
