# AGENT TASK — future rank-affinity single-copy placement

Checkpoint: `6a8127c`.

Read [PLAN.md](PLAN.md).

Implement this as a small extension/reuse of the existing rho=0 cache and
traffic semantics. Do not create a second incompatible cache model.

Bounded sequence:

1. reproduce B8 F exactly and stop on any mismatch;
2. add O0 current-event exact peer-byte owner selection;
3. add OH1/OH2/OH4/OHremaining offline owner scoring;
4. enforce globally unique residency: no duplicates and no migration;
5. run exactly six policies on B8/B16/B32 = 18 CPU cells;
6. publish raw H2D/peer coordinates and owner/eviction trajectories;
7. separate F->O0 current-placement gain from O0->OH future-affinity gain;
8. compare only B8 against the committed old F/rho=.125/K/rho=.5/C frontier;
9. run tests/validation, commit compact results, stop.

Constraints:
- CPU only;
- no model, Torch, CUDA or NCCL;
- no new trace capture;
- no R8;
- no cache-ratio sweep;
- no replication or migration;
- no substitution;
- do not alter/stop the eight resident-model GPU workers;
- no physical timing after the CPU result.

If a secondary B16/B32 raw receipt is missing, record it and continue with the
available existing traces; do not generate a new model trace.
