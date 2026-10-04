# Transport-stack remeasurement (2026-10-04)

This packet fixes two implementation issues before interpreting BR/CA TPOT.

## A — CoSLoT-style pinned H2D
Primary runs use a bounded two-expert pinned CPU staging pool, dedicated H2D
CUDA stream, async pinned->GPU copy, fetch events before expert compute, and
compute events before cache-slot overwrite. The full model is not pinned.

## B — active-peer communication
The exact same frozen schedule is executed with:
- current: token->rank, 3 all_to_all_single rounds/layer;
- coslot: fused token-expert forward A2A + return A2A, 2 rounds/layer;
- coslot-active: same fused payload but grouped isend/irecv only for non-empty
  rank pairs. Local-only rounds skip NCCL communication entirely.

Env1 can physically use P2P/IPC. Env2 still uses the active-peer API graph, but
NCCL_P2P_DISABLE=1 forces the physical transport through SHM.

## Missing PLAN is handled automatically
No PLAN path is required. The runner first searches for PASS+validated PLANs.
If R/BR or R/CA is missing, it creates a current/pageable PLAN outside timing,
runs validate_env_offload_plan.py, freezes it, then reuses it for every
transport/H2D measurement.

Run:
```bash
cd mgo_v2
PYTHONPATH="$PWD:$PWD/scripts" python scripts/run_coslot_comm_remeasure.py \
  --cell R --environment env1 --policies BR CA --repeats 3
```
Then repeat env2 only after env1 timing is stable.

COUNTERS records A2A calls, grouped P2P batches, send/recv ops, zero-remote
rounds, active-peer degree, H2D bytes, activation peer bytes and wire bytes.
