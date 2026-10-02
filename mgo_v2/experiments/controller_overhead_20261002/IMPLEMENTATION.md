# Decision-equivalent controller implementation

The baseline modules and `benchmark_model.py` remain byte-for-byte unchanged.
New paths are opt-in through `enable_local_optimization(controller)` and
`SinglePlanner(controller, torch, dist)`. They are installed only on a fresh
controller after normal model warmup. The original placement/oracle packet is
completed and committed before activation of this separate stage.

## C1: indexed state and exact array ranks

`IndexedCache` delegates authoritative place/evict/touch operations to the
original cache. It maintains numeric slot/key/last-use arrays and per-layer
resident sets. Resident views are immutable and reused until that layer changes.
Every successful placement/eviction updates the original Coverage neighborhood
counts; only dirty layers recompute their damage vector before victim selection.
There is no second independent placement or cache manager.

`IndexedHistory` calls the original rolling-window update and caches its exact
float64-mean-to-float32 gate scores. Victim candidates exclude exactly the same
pinned keys. Array midranks are `2 * lower_count + tie_count - 1`; selection uses
the unchanged Coverage score, last-used tick and lexicographic expert key. The
integer flat key preserves `(layer, expert)` ordering. Candidate visits are not
reduced; Python list/dictionary construction, tuple scoring and Python sorting
are removed from the Coverage path.

Full consistency scans are gated by the opt-in cache's debug flag. Offline
validation checks the authoritative cache, slot/last-use views, resident sets,
Coverage counts, pending-dirty semantics and computed loss vectors. It does not
flush lazy views or dirty state outside timing and thereby move work out of the
next measured plan. The normal final physical-cache check remains active.

## C2: one planner and mirrored decisions

All ranks gather the original global route metadata. Rank 0 executes C1 and
encodes a fixed int32 payload: version/status/layer/tick/policy/shape/quotas and
nine fields per expert (substitution target/tier/flags, execution/admission owner,
slot, victim layer/expert and operation order). With this model the payload is
4,656 bytes per event. One NCCL broadcast transfers it. Preallocated pinned CPU
and CUDA buffers avoid dynamic collective shapes.

Followers validate the complete payload against current residency before cache
mutation. They reconstruct effective weights with the original merge function,
consume and verify O0's frozen record where applicable, update history once and
apply the exact operations. They never choose a substitute, admission or victim.
The C2 random policy's follower RNG state is intentionally unused; this is not a
planner-failover implementation. A leader planning/encoding exception broadcasts
a failure header so peers fail rather than waiting for a decision indefinitely.

Diagnostic transport spans separate planner compute, encode, broadcast/copies/
synchronization and follower apply. Follower collective waits can include leader
CPU planning; they are not pure network latency. Payload and modeled peer bytes
exclude NCCL protocol overhead. The short GPU profile reconciles payload copies
and reports added NCCL work separately from unchanged expert fetches/GEMMs.

## Correctness and measurement boundary

The unit suite checks exact tied ranks, gate rounding, random asymmetric
similarities, both P0/P1 and frozen O0, full state equality, payload round-trip,
malformed-payload rejection and diagnostics on/off parity. Fresh C0 GPU runs
capture raw routes outside the generation timer. Full CPU replay then compares
C0, C1 and the follower codec event by event, including complete substitution,
admission, execution operations, effective weights, owners/slots/timestamps and
history. C1 GPU runs precede C2, with all physical events and tokens checked
against the completed original packet on every rank.

Primary timing retains the same compact CPU evidence/row-count capture for all
variants. Hashing, JSON/pickle writes, offline checks and token parity are outside
the benchmark's generation timer. Only the runtime's existing total controller
clock remains; detailed CPU timers and GPU profiling are separate. The owner
reduced this stage to nine full generations and two short posthoc P1 profiles.
There is one sample per cell, no controller-order counterbalancing and no B4/B16
expansion. Small performance differences are not treated as established gains.
