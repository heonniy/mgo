# PLAN — Rank-local expert reuse and selective-replication break-even

Date: 2026-10-03  
Status: prospective, owner-authorized CPU-only headroom study.

## Why this experiment

The existing replica-budget screen showed a real fetch/communication frontier,
but the immediate greedy replicas are expensive. At B8 decode:

- F (rho=0): H2D 154.995 GiB, peer 319.453 MiB;
- K (rho=.25): H2D 219.076 GiB, peer 185.297 MiB.

Thus K spends roughly +64 GiB of expert H2D to save only ~134 MiB of peer
activation traffic.

Before designing a controller, test whether this inefficiency comes from
replicating the wrong `(layer, expert, rank)` pairs. A useful selective policy
requires **rank-local temporal reuse**: after a remote service, the same rank
must demand the same expert again before the replica is evicted.

## Scope

CPU only. No model, CUDA, NCCL, new trace capture, F/K physical run, R8 run,
substitution, placement redesign or transport tuning.

Do not stop or alter the owner's eight resident-model GPU workers. The analysis
must set `CUDA_VISIBLE_DEVICES=''`, BLAS/OMP threads=1 and stay within a 4-GiB
process address-space bound.

## Frozen inputs

Primary input:

- existing exact-only B8/global32 capture used by the replica Pareto study;
- 30% global expert-cache budget;
- R4;
- same `ROW_BYTES=4096`, `EXPERT_BYTES=9 MiB`;
- same LRU, active protection, first-copy rule and traffic accounting as
  `replica_pareto_cpu.py`.

Secondary consistency inputs:

- existing B16/global64 exact-only capture;
- existing B32/global128 exact-only capture.

B4 is not required for the first decision because the old F/K/C frontier is B8.
Do not add a new capture if any secondary raw receipt is unavailable; report the
missing secondary input and complete B8.

Verify every raw receipt size/SHA256 before use. Rank 0's capture contains the
global `origin_ranks` and `raw_selected_experts`; verify cross-rank hashes
from the committed summaries before trusting it.

## Important horizon limitation

The current captures contain only eight decode forwards. Therefore the allowed
lookahead horizons are:

```text
H = {1, 2, 4, remaining-to-end (max 8 decode steps)}
```

Do **not** report H=16 from these traces.

A horizon is measured in decode steps for the same layer, while intervening
layers still affect cache state in capacity-aware replay.

---

## Stage R0 — reproduce the no-replica F baseline

Reuse the independently checked rho=0 replay semantics.

For every decode layer event, save:

- origin rank;
- selected exact expert;
- execution destination under F;
- whether service is local or remote;
- exact dispatch/combine count matrices;
- physical cache state before and after the event.

The B8 aggregate must match the committed F decode totals exactly before any
reuse statistic is accepted:

```text
total_fetches              = 17,635
expert_h2d_bytes           = 166,424,739,840
peer_activation_bytes      = 334,970,880
remote_token_rank_pairs    = 23,179
```

Any mismatch is a hard stop.

---

## Stage R1 — characterize rank-local temporal reuse

The key is `(layer, expert, requesting_rank)`, not global expert popularity.

For every remote F service occurrence, compute:

1. remote expert-route rows at the current decode step;
2. exact marginal peer bytes that would disappear if that expert had a local
   copy on the requesting rank for this event:
   - one combine row per moved expert route;
   - a dispatch row only when moving the candidate removes the last route from
     that token to the old remote destination;
3. next remote reuse distance in decode steps;
4. cumulative same-pair marginal peer bytes over H={1,2,4,remaining};
5. number of future decode steps with positive same-pair demand;
6. consecutive positive-demand burst length.

Report traffic concentration by `(layer, expert, rank)`:

- top 1%, 5%, 10%, 20% share of remote expert routes;
- same shares of exact marginal peer bytes;
- Lorenz/Gini-style concentration statistic;
- fraction of remote bytes coming from pairs that recur within 1/2/4 steps.

This stage answers:

> Is remote traffic diffuse, or is a small set of rank-local expert pairs
> repeatedly responsible for it?

---

## Stage R2 — persistence-only replica upper bound

Construct a deliberately optimistic upper bound before modeling eviction.

For each first remote occurrence of a `(layer, expert, rank)` pair:

- pretend one replica can be fetched once;
- it persists for the chosen horizon without consuming a cache slot;
- all future same-pair routes in that horizon become local;
- count exact dispatch+combine peer-byte savings from the frozen F destinations.

Report per candidate:

```text
future_peer_bytes_saved(H)
replica_fetch_bytes = 9 MiB
reuse_steps
reuse_routes
first_break_even_horizon_by_bytes (if any)
```

The peer-byte/H2D-byte ratio is a **transport-independent accounting ratio**,
not a time ratio. Do not claim that 1 byte of H2D equals 1 byte of peer traffic.

This stage is only an upper bound: if even the persistence-only oracle shows
little repeated traffic, stop without implementing a controller.

### Headroom gate

Continue to capacity-aware replay only if, for B8, at least one of these holds:

- top 10% of `(layer,expert,rank)` pairs account for >=40% of remote marginal
  peer bytes; or
- >=25% of remote marginal peer bytes belong to pairs that recur within the
  next 4 decode steps.

Otherwise label `LOW_RANK_LOCAL_REUSE` and stop after R2.

---

## Stage R3 — capacity-aware lookahead selective replay

This is a future-aware **diagnostic policy**, not an implementable online
controller and not claimed to be globally optimal.

Keep exactly the same physical cache slots and LRU semantics as the existing
replica replay.

At each event:

1. perform mandatory exact admissions exactly as F;
2. enumerate remote `(expert, requesting_rank)` candidates;
3. score each candidate by its precomputed future exact marginal peer-byte
   saving over horizon H;
4. account for a potential victim:
   - active experts are never evicted;
   - free slot first;
   - otherwise the normal legal LRU victim;
   - add a one-fetch victim penalty if that unique victim is demanded again
     within H and has no surviving copy;
5. admit only candidates whose future score is positive after the chosen
   admission threshold;
6. execute the event and let all resulting evictions, reloads, local services
   and LRU touches affect subsequent events normally.

Use only these horizons:

```text
H = {1, 2, 4, remaining}
```

For each H, generate a small cost-independent Pareto by varying a fixed
minimum future peer saving per replica admission:

```text
threshold = {0, 64 KiB, 256 KiB, 1 MiB}
```

16 CPU cells per batch maximum. No post-hoc thresholds.

Report actual trace totals:

- first/reload/replica fetch counts;
- H2D bytes;
- peer activation bytes;
- remote token-rank pairs;
- duplicate-slot mean/peak;
- unique resident experts;
- number of replica admissions;
- fraction of replicas reused before eviction;
- replica lifetime;
- peer bytes saved per replica fetch;
- victim-induced reload count.

---

## Stage R4 — compare against the existing frontier

For B8, overlay:

```text
F = rho 0
K = rho .25
C = rho .75
lookahead-selective cells
```

in the same plane:

```text
x = decode peer activation bytes
y = decode expert H2D bytes
```

Primary research question:

> Does future rank-local reuse awareness create a point that dominates K or
> materially bends the old F-K-C frontier toward the lower-left?

Predeclare:

### STRONG_HEADROOM

At least one selective cell satisfies either:

- peer bytes <= K and H2D <= 0.75 * K H2D; or
- H2D <= F + 0.25*(K-F) while achieving >=50% of F->K peer-byte reduction.

### MODEST_HEADROOM

A selective point dominates at least one old nonzero-rho point but misses the
STRONG_HEADROOM rule.

### NO_HEADROOM

No selective point dominates any old nonzero-rho point.

These labels are CPU accounting only; none imply E2E speedup.

For B16/B32, report the same reuse statistics and selective Pareto coordinates,
but do not invent old F/K/C references if they do not exist.

---

## What this experiment can establish

A positive result supports the problem formulation:

```text
default: remote exact execution
exception: replicate only rank-local experts with enough future reuse
```

It would motivate an online predictor based on recent per-rank expert demand.

A negative result is also decisive: if rank-local reuse is weak, selective
replication is unlikely to justify a controller, and the project should return
to unique-coverage/fetch optimization rather than forcing a communication
objective.

## Required outputs

Under this experiment directory:

- `reuse_pair_summary.csv/json`
- `reuse_horizon_summary.csv/json`
- `persistence_upper_bound.csv/json`
- `selective_replay_points.csv/json` (only if R2 gate passes)
- `frontier_comparison.csv/json`
- `RESULTS.md`
- `validation.json`

Raw external capture receipts remain outside Git; commit only compact derived
outputs and their provenance hashes.

## Stop condition

Commit after R0-R2 if the headroom gate fails.

If it passes, run R3-R4, commit results, and stop for owner review.

No GPU follow-up is automatically authorized.
