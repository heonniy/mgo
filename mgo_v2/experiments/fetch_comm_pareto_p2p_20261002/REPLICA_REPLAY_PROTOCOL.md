# Frozen CPU replay conventions

This implements `REPLICA_PARETO_SCREEN.md` only. No runtime controller change,
model import, GPU access, NCCL characterization, or physical F/K/C follows it.
The protocol is committed before observing the five-budget sweep.

- Input: hash-verified exact-only capture receipts from `ed7f82b`; replay rank 0's
  complete global raw routes, all 48 prefill events followed by 384 decode events.
  Each rho starts empty. Capture owner choices are deliberately not reused.
- Capacity: floor(48 × 128 × .30) = 1843 physical slots, split [461,461,461,460].
  Rho is exactly [0,.125,.25,.5,.75], with no additional trial budgets.
- Mandatory global misses are admitted in increasing expert ID order. Target is
  maximum origin demand, then smallest rank. All active experts' copies are
  protected throughout the event. Mandatory capacity is preflighted on every
  rank before any mutation; an impossible event fails without changing state.
- Lowest free slot is used first. LRU is per physical copy: admission sets its
  timestamp, and actual route service touches it at event end. Timestamps are
  event indices; equal-time victims tie by (layer, expert). Remote service
  touches the serving copy, and unused replicas are not touched.
- Primary stays the first copy while resident. Its eviction promotes the
  smallest surviving rank. Local copies always take precedence when serving.
- Replicas are admitted after mandatory copies and before service. A candidate
  is legal only if duplicate_slots < floor(rho × 1843) and its target has a free
  slot or inactive victim. At a full cap, stop even if evicting an inactive
  duplicate might make room. This follows the plan's "budget is full" stop.
- Exact candidate byte saving is computed incrementally, then recomputed after
  each chosen replica: one 4096-byte combine row per moved exact route plus one
  4096-byte dispatch row for each affected token with no other expert remaining
  on the previous remote destination. Tests compare this against rebuilding
  the entire dispatch/combine matrices for every candidate on synthetic traces.
  Largest positive saving wins, ties by expert ID then rank. Inactive unique
  experts may be evicted. Other active destinations cannot change mid-event.
- Dispatch is one hidden row per token/destination rank; combine is one row per
  selected expert. Self edges are omitted from peer bytes. Metadata, weights,
  cache copies and collectives are not activation bytes. Receive matrices are
  counted separately and must transpose-match sends. Every route has exactly
  one resident execution destination.
- Fetch categories are disjoint: first-copy means first-ever global admission,
  reload means global miss of a previously fetched expert, replica means adding
  a copy of a currently resident expert. Every copy counts 9 MiB H2D.
- Local exact-hit fraction counts routes with a local copy at event entrance.
  Global-resident remote-service fraction counts final remote routes whose
  serving copy existed at entrance. Newly loaded copies are not hits. Also
  publish final local-service and entrance global-hit fractions for clarity.
  Fractions are weighted by raw expert routes; occupancy means are unweighted
  means of post-event state. Peak duplicates includes entrance/intermediate
  state. Duplicate fractions divide by 1843; unique coverage counts (layer,expert).
- Report prefill, decode and full trace separately. Only decode bytes select
  the Pareto frontier. Dominance means no worse on both axes and strictly better
  on one. F minimizes H2D (tie peer bytes, rho); C minimizes peer bytes (tie H2D,
  rho). Equal coordinates are not separate useful shapes. K requires at least
  three distinct nondominated coordinates and a positive inward distance from
  the normalized F--C chord; maximize that distance, tie smallest rho.
- Endpoint changes divide by F's decode totals: (C.H2D−F.H2D)/F.H2D and
  (F.peer−C.peer)/F.peer. GO requires three useful nondominated rho settings and
  >=10% on both axes, and still stops for owner review without physical runs.
- Independently implemented slot-array rho=0 replay must match all 432 events'
  destinations, traffic matrices, fetch classes and full cache/LRU/seen state.
  Policy unit tests also cover active protection, atomic failure, tie breaks,
  primary promotion, unique-copy eviction, reloads and randomized exact savings.
- Resource bound: one CPU process, BLAS/OMP threads=1, CUDA visibility empty,
  address-space hard limit 2 GiB. Outputs contain process peak RSS and elapsed
  time, not GPU performance measurements. Each rho runs once.

Reproduce from the repository root with the NumPy environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/hwlee/sub-moe/phase01/.venv/bin/python mgo_v2/scripts/test_replica_pareto_cpu.py -v
/home/hwlee/sub-moe/phase01/.venv/bin/python mgo_v2/scripts/run_replica_pareto_screen.py
```
