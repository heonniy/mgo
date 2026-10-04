# Transport-stack remeasurement (2026-10-04)

This packet normalizes expert H2D and tests three GPU communication stacks.

## A — CoSLoT-style pinned H2D
Primary runs use a bounded two-expert pinned CPU staging pool, dedicated H2D
CUDA stream, async pinned->GPU copy, fetch events before expert compute, and
compute events before cache-slot overwrite. The full model is not pinned.

## B — active-peer communication
The exact same frozen schedule is executed with:
- current: token->rank, 3 all_to_all_single rounds/layer;
- coslot: fused token-expert forward A2A + return A2A, 2 rounds/layer;
- coslot-active: the same fused payload with grouped isend/irecv only for
  non-empty rank pairs. Local-only rounds skip NCCL communication.

Env1 can physically use P2P/IPC. Env2 uses the same logical active-peer graph,
but NCCL_P2P_DISABLE=1 forces the physical transport through SHM.

## C — 60% cache, substitution OFF
The requested follow-up is R/MATH, R4, local B64, Gate eviction, cache=60%,
with substitution disabled.

The prior CPU replay for the matching substitution-OFF configuration showed:
- c30: exact global hit ~59.9%, residual miss ~40.1%, H2D ~7.16 TiB;
- c60: exact global hit ~86.5%, residual miss ~13.5%, H2D ~3.26 TiB;
- BR peer traffic stayed ~104.5 GiB.

This arm therefore asks whether communication becomes more important once
offloading pressure is much lower, without substitution confounding the hit
rate.

## PLAN handling
No PLAN path is required. PLAN discovery is keyed by both cache ratio and
substitution mode. If the requested c60/s0 BR or CA PLAN is missing, the runner
creates a current/pageable reference PLAN outside timing, validates it, then
reuses that exact schedule for every transport.

A c30 or substitution-ON PLAN cannot be silently reused for c60/s0.

Run the requested arm:
```bash
cd mgo_v2
PYTHONPATH="$PWD:$PWD/scripts" python scripts/run_coslot_comm_remeasure.py \
  --cell R --environment env1 --policies BR CA \
  --cache-ratios 0.60 --substitution off --repeats 3
```

For a direct cache-pressure comparison:
```bash
PYTHONPATH="$PWD:$PWD/scripts" python scripts/run_coslot_comm_remeasure.py \
  --cell R --environment env1 --policies BR CA \
  --cache-ratios 0.30 0.60 --substitution off --repeats 3
```

COUNTERS record A2A calls, grouped P2P batches, send/recv ops, zero-remote
rounds, active-peer degree, H2D bytes, activation peer bytes and wire bytes.
