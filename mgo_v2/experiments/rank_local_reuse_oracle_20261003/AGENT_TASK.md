# AGENT TASK — rank-local reuse / selective replication

Current owner checkpoint: `02eeb59`.

Read [PLAN.md](PLAN.md) and reuse the existing CPU replay code rather than
creating a second cache semantics.

Do this bounded sequence:

1. hash-verify the existing B8 exact capture and reproduce F exactly;
2. derive remote `(layer,expert,requesting-rank)` temporal reuse;
3. compute exact per-event marginal peer-byte savings and H={1,2,4,remaining}
   future reuse;
4. publish the persistence-only upper bound;
5. apply the predeclared R2 reuse gate;
6. only if it passes, run the 4 horizons x 4 fixed-threshold capacity-aware
   selective replays;
7. compare their B8 H2D/peer coordinates against committed F/K/C;
8. optionally process existing B16/B32 as consistency traces; missing secondary
   raw files do not justify a new model capture;
9. commit compact results and stop.

Constraints:

- CPU only, `CUDA_VISIBLE_DEVICES=''`;
- one process, BLAS/OMP threads=1;
- <=4 GiB address space;
- no GPU worker interruption;
- no model/NCCL;
- no new trace capture;
- no R8;
- no substitution;
- no new physical F/K/C;
- no timing claim from peer bytes.

Implement unit tests for exact marginal dispatch/combine savings, reuse-distance
calculation, victim penalty and deterministic ties before running the trace.
