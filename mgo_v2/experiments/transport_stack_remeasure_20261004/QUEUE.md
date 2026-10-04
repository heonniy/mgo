# Active queue: c60/s0 only

The owner explicitly removed arm 1 (c30/substitution ON). Its queue dispatcher
and in-progress CA PLAN were stopped before any timed measurement. Preserve
its completed BR PLAN and cancellation receipts, but never resume c30/s1.

Only arm 2 remains: c60/substitution OFF, R/MATH/R4/local-B64/decode256,
GPUs0,1,4,5, pinned H2D, BR/CA, current/coslot/coslot-active.
Each condition uses COMPILE x1, MEASURE x3, COUNTERS x1.
Env1 first; Env2 only after all six Env1 conditions pass the 5% spread gate.
Create and validate c60/s0 reference PLANs independently; do not reuse c30/s1.
Commit checkpoints and restore resident-model workers after completion/failure.
No other experiment is queued.
