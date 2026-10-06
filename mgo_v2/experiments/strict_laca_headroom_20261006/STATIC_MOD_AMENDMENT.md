# Static ownership baseline — execution on hold

Owner follow-up: postpone execution. Waiting queue stopped before any static
GPU run or full trace replay. Implementation and CPU unit checks are retained.
Do not resume automatically; earlier execution authorization is withdrawn.

Add STATIC_MOD to all eight selected final conditions: R4/R8 x localB16/B64
x input256/512. For every missing expert e, fetch to logical rank e % world.
For R4 the logical-to-physical map is0,1,4,5; for R8 it is0..7. Expert IDs are
layer-local0..127. With128 divisible by4/8, using the full layer-expert key
would produce the same mapping. Ownership remains fixed on hits and refetches.

Preserve C30 capacities, Gate eviction, cold prefill start, policy state through
32 decode steps, BF16, frozen token/routes and strict serialized barriers.
No substitution, replication, speculative prefetch, seed reselection or new
trace capture. Static deliberately does not enforce BR's balanced miss-count
quota. Record physical copies per rank and distinguish changed H2D work from
pure placement redistribution.

Do not modify runtime sources used by active three-policy measurements.
The static implementation is isolated; it copies the existing legacy policy
accounting/eviction and replaces only the miss destination with modulo rank.
CPU checks cover skewed misses, resident hits, eviction/refetch for R4/R8.
Replay all1584 events to build independent static physical-copy/state proofs;
verify every fetch and resident expert belongs to its fixed owner. GPU warm
and measured tokens must match the saved BR33-output reference.

After the current decode32 queue completes, prepare static CPU proofs (up to
8 processes), then run eight static GPU cells sequentially. One correctness
warmup and two measured repeats per cell; third only if the maximum metric
relative difference is2-5%;>5% unstable without extra repeats. Metrics are
TTFT, wall TPOT, E2E. No diagnostic passes. Do not rerun existing policies.

Static is added sequentially, not interleaved with the earlier BR/candidates.
The combined report explicitly labels possible session drift and retains
all original estimates. The existing favorable seeds were selected for
LA_CA_NEAR vs BR; they are not a neutral dataset-average static comparison.
