# PLAN — Hot-expert local replication break-even

Date: 2026-10-03  
Status: **QUEUED**; do not preempt the active single-copy placement study.

## Research question

The original fetch-vs-communication hypothesis was:

```text
fast interconnect  -> remote service is cheap -> little replication
slow interconnect  -> remote service is expensive -> more replication
```

The fixed replica-budget experiments showed a Pareto frontier, but naive
replication paid far too much H2D/cache capacity for the peer traffic it saved.

This packet tests a narrower policy hypothesis:

> A rank should replicate only an expert with sufficiently high **current
> rank-local token demand**; colder expert demand should remain remote.

Define, for layer `l`, expert `e`, requesting rank `r`, decode step `t`:

```text
n(l,e,r,t) = number of local decode tokens on rank r selecting expert e.
```

The desired rule is conceptually:

```text
replicate if measured remote-service critical-path cost
             >
             expert H2D + cache/victim cost
otherwise remote serve
```

This is a characterization first, not a final online controller.

---

## Important interpretation boundary

The two transport conditions on the HGX machine are:

- **T0-IPC**: direct P2P/IPC over NVSwitch;
- **R3-SHM**: direct P2P disabled, NCCL SHM/direct/direct.

R3 is a synthetic slow-communication stress condition. It is **not** a
physical PCIe-only server and must not be labeled as one.

The experiment may establish that the hotness crossover changes between these
two controlled conditions. A real PCIe-only claim requires separate hardware
validation later.

---

# Stage H0 — existing-trace hotness characterization (CPU only)

Do not run a new model first.

Use existing exact-only R4/cache30 traces:

```text
local B8  / global B32
local B16 / global B64
local B32 / global B128
```

For every decode `(layer, expert, requesting-rank, step)` with positive demand,
report:

- `n`: current local token demand;
- whether the expert is local/resident/remote under rho=0 F;
- exact marginal dispatch bytes saved if that pair became local **for this
  current event**;
- exact marginal combine bytes saved;
- exact total peer bytes saved;
- current serving owner;
- whether another selected expert for the same token shares that remote owner
  (dispatch-row sharing).

Use the existing traffic semantics exactly: one dispatch row per
token/destination rank, one combine row per selected expert, 4096 B per row.

## Required distributions

Per batch:

- `n` p50/p90/p95/p99/max;
- remote-only `n` p50/p90/p95/p99/max;
- fraction of remote routes and peer bytes attributable to demand thresholds:
  `n >= {1,2,4,8,16,32,64,128}` where populated;
- top 1/5/10/20% rank-local hot pairs by demand and their peer-byte share;
- relationship between `n` and exact marginal peer-byte saving;
- number of candidates for which one current-event local copy saves at least
  64/128/256/512 KiB or 1/2/4/8 MiB.

Do not convert peer bytes to latency in H0.

## Sanity bound

For each candidate, verify:

```text
exact current peer saving <= 2 * n * 4096 bytes
```

because at most one dispatch row and one combine row per moved route can vanish.

Compare observed hotness scaling B8 -> B16 -> B32. This answers whether larger
local batch actually creates a high-demand tail rather than merely scaling
aggregate traffic.

---

# Stage H1 — isolated break-even microcost calibration

Run H1 only **after the active single-copy placement packet has committed and
stopped**.

No model is loaded.

Use physical GPUs 0,1,4,5 and the stable transport definitions already
validated in the repository.

## H1-A remote-service pair cost

Construct a deterministic communication-only pattern representing one
origin-rank / one remote-owner expert service with `n` BF16 hidden rows.

Use:

```text
n = {1,2,4,8,16,32,64,128,256,512}
row = 4096 B
```

For each n, time the **dispatch + combine pair** under T0-IPC and R3-SHM.

Requirements:

- payload validation outside timing;
- 10 warmups;
- 30 timed samples;
- counter-ordered transport passes;
- median and p90 max-rank CUDA interval;
- no INFO logging in timing;
- no model/controller/cache work.

This calibration is deliberately pair-local. It estimates the critical-path
price of serving a hot rank-local demand remotely; it is not a full MoE
all-to-all throughput benchmark.

## H1-B one expert H2D

Measure one 9-MiB CPU->GPU expert-shaped copy using the existing pinned-memory
path:

- isolated one-rank H2D;
- four ranks issuing one H2D concurrently.

10 warmups + 30 samples, median/p90.

The four-rank case is required because replication decisions can create
simultaneous fetch pressure.

## Break-even definitions

For each transport and H2D condition, define a **measured isolated crossover**:

```text
n*_isolated =
smallest measured n for which
remote_pair_median(n) >= expert_H2D_median
```

Also compute a conservative p90 crossover:

```text
remote_pair_p90(n) >= expert_H2D_p90.
```

If no measured n crosses, report `>512`; do not extrapolate a numerical
threshold from a weak linear fit.

These thresholds exclude cache-victim cost, so they are optimistic lower bounds
on the demand required to justify a replica.

---

# Stage H2 — trace coverage at the measured threshold

Map the H1 thresholds back onto B8/B16/B32 traces.

For each transport/H2D condition report:

- fraction of remote candidate occurrences with `n >= n*`;
- fraction of remote expert routes covered;
- fraction of exact marginal peer bytes covered;
- number of distinct `(layer,expert,rank)` pairs that cross;
- layer distribution of crossing events.

This directly answers:

> At current local batches, how often is an expert actually hot enough to make
> one local replica plausible under the isolated cost bound?

If `n*>512`, all current traces are automatically below the measured range;
report that fact rather than inventing coverage.

---

# Stage H3 — threshold policy replay, only if H2 has candidates

Run a capacity-aware CPU replay only for thresholds obtained from H1; do not
tune thresholds from trace outcomes.

Policy:

- rho=0 F semantics by default;
- on a remote `(expert,rank)` demand with `n >= n*`, a local replica may be
  admitted if there is a legal free slot/inactive victim;
- active copies protected;
- normal per-copy LRU;
- one 9-MiB H2D for every replica;
- victim reloads counted normally;
- remote service for `n<n*`;
- no future predictor;
- no migration;
- no substitution.

Run at most:

```text
3 batches x
{T0-median threshold, R3-median threshold,
 T0-p90 threshold, R3-p90 threshold}
= 12 CPU replay cells
```

Deduplicate identical thresholds.

Report raw:
- H2D bytes/fetches/reloads;
- peer activation bytes;
- replica admissions;
- local-service fraction;
- victim-induced reloads;
- unique coverage;
- replica reuse before eviction.

Compare B8 against old F/rho=.125/K/rho=.5/C raw frontier, but make no latency
claim from byte coordinates.

---

# Stage H4 — decide whether larger local batch is worth capturing

Do **not** automatically run B64/B128.

Use the measured `n*` and B32 hotness distribution.

### LARGE_BATCH_CAPTURE_WORTHWHILE

Authorize a separate owner-reviewed B64 capture proposal only if either:

1. B32 already has nonzero remote-route coverage at the optimistic R3 median
   threshold; or
2. B32 max remote demand is at least 0.5 * that threshold and observed
   B8->B16->B32 max/p99 demand grows monotonically.

A B128 capture is never automatic. It requires review after B64.

### CURRENT_BATCH_TOO_COLD

If B32 max demand is below 0.5 * the optimistic R3 threshold, stop. State that
the current batch regime is too cold for current-step hotness replication under
the measured isolated lower bound.

### NO_TRANSPORT_CROSSOVER

If neither T0 nor R3 crosses the 9-MiB H2D cost by n=512, stop without larger
batch captures. Do not extrapolate.

---

# Expected research interpretations

The useful outcome is not "replication always wins."

A strong result would look like:

```text
n*_R3 < n*_T0
```

and a nontrivial fraction of large-batch remote traffic lies above
`n*_R3` but below `n*_T0`.

That would support the original qualitative hypothesis in a sharper form:

> expensive interconnect lowers the rank-local demand threshold at which local
> replication becomes worthwhile.

A negative result is equally useful: if even R3 requires unrealistically hot
rank-local demand, fixed/threshold replication should not be a core method on
this model/system.

---

# Resource and correctness gates

- H0/H2/H3 CPU-only: `CUDA_VISIBLE_DEVICES=''`, one process, BLAS/OMP=1,
  <=4 GiB address-space;
- H1 only uses GPUs 0,1,4,5 after active research work has stopped;
- pause/resume only the owner's resident-model workers according to the existing
  documented guard; never kill other users' jobs;
- all source receipt SHA256 hashes verified;
- B8 F exact reproduction required before H3;
- no substitution;
- no R8;
- cache ratio fixed 30%;
- no NCCL tuning;
- no NVLink bandwidth-mode operations;
- no Nsight;
- no physical F/K/C model timing.

# Required outputs

- `hotness_distribution.csv/json`
- `hotness_peer_saving.csv/json`
- `break_even_microcost.csv/json` (if H1 runs)
- `threshold_coverage.csv/json`
- `threshold_policy_points.csv/json` (if H3 runs)
- `RESULTS.md`
- `validation.json`

Commit each stage boundary and stop for owner review after H4.
