# PLAN — phase-aware R8/B128-B256 placement + realizable replication

## 1. Questions

At R8 and 60% global expert cache:

1. Does increasing local batch from B128 to B256 increase the benefit of
   hot-expert replication and/or load-aware miss placement?
2. On a communication-worst random baseline, is CA still the best correction,
   or does replication / load-aware placement provide more headroom?
3. On a compute-load-worst random baseline, is replica+load-aware placement
   substantially better than CA or replica+CA?
4. Once real cache slots, future reloads and replica D2D creation are counted,
   how much of the earlier free-replica oracle survives?
5. Under the planned GPU order
   metadata -> decision -> dispatch -> H2D/D2D transfer phase -> fetch barrier
   -> expert compute -> combine,
   does the structural improvement translate into phase-time headroom?
6. Does the same mechanism materially affect prefill, or is its useful claim
   decode/TPOT only?

All results in this packet are CPU replay / calibrated phase-model results.
They are not physical TPOT/E2E measurements.

## 2. Frozen system

- Qwen3-30B-A3B exact routes;
- R=8;
- local B in {128, 256};
- global cache = 60%;
- Gate W128 eviction;
- substitution OFF;
- prefill + 256 decode steps;
- top-k=8;
- exact 2048-request MATH and ShareGPT pools already captured.

B128 uses 1024 requests and therefore has sample-subset freedom.
B256 consumes all 2048 requests. For B256 the sample seed is fixed/redundant;
only DP/rank order and random placement seed are searched.

## 3. Six policies

Mandatory exact misses always use the same balanced per-event fetch quota:
each rank receives floor(M/R) or ceil(M/R) new experts. Thus no policy can win
by simply pushing more PCIe misses onto another rank.

### BR
Balanced-random mapping of mandatory miss experts to rank quota slots.

### CA
Same balanced quota slots, Hungarian assignment maximizing current local
expert-route demand (communication-aware placement).

### LA
Same balanced quota slots, online load-aware assignment. Current resident
expert load is estimated first; miss experts are processed by descending
current demand and assigned to a remaining rank quota slot that minimizes the
resulting maximum rank expert-row load. Ties prefer lower load then more local
demand. No future routes are read.

### BR+REP / CA+REP / LA+REP
The respective mandatory-miss policy plus realizable persistent hot-expert
replication.

At most two copies of one expert and at most one new replica per layer-event
are allowed.

## 4. Realizable replica semantics

Replication is evaluated **after mandatory misses have been placed** for the
current event. Therefore a replica candidate can be:

- an expert that was already resident before the event; or
- a current miss expert that has just received its mandatory H2D copy.

The decision uses current routing and cache state only.

A candidate second copy must:
- consume one real destination-rank cache slot;
- use the same Gate W128 victim semantics;
- respect a global duplicate-copy budget;
- exceed the configured current-event hotness threshold;
- reduce current maximum-rank expert rows by at least the configured threshold.

The chosen replica persists until normal eviction. If its victim was the last
copy of another expert, any later use naturally becomes a real H2D reload.

Execution across two copies is load-aware: expert rows are split to reduce
rank load, with origin-local service used as a tie-breaker.

## 5. Replica transfer dependency and overlap

Every new replica is one expert-sized GPU-to-GPU D2D copy.

### Resident source
If the source expert was already resident, D2D may start at the beginning of
the transfer phase and overlap with mandatory H2D traffic.

### Current-miss source
If the replicated expert is itself a miss:

1. its mandatory H2D copy is prioritized first on the source rank;
2. as soon as that expert H2D finishes, its D2D copy can start;
3. the source rank and all other ranks continue fetching their remaining misses
   concurrently with that D2D copy.

This implements the owner's requested model:

```text
dispatch
   |
   +-- rank A: H2D(Ehot) ---- H2D(Emiss2) ---- H2D(Emiss3)
                   \
                    +-- D2D(Ehot -> replica rank)
```

The transfer barrier releases expert compute only after all mandatory H2D and
the replica D2D have completed.

For every event report both:
- `overlap_transfer_ms`: D2D overlaps remaining PCIe H2D as above;
- `serial_transfer_ms`: counterfactual H2D then D2D, no overlap.

This isolates the value of the overlap assumption.

## 6. Replica policy grid

Predeclare a small headroom/safety grid instead of hand-tuning one threshold:

- hotness threshold = {1.0, 1.5, 2.0} * local batch rows;
- minimum current max-rank reduction = {2%, 5%};
- duplicate budget = {2%, 5%} of total cache slots.

12 configurations per base placement.

For each base policy report:
1. maximum structural compute-balance improvement;
2. best policy with H2D growth <=2%;
3. best calibrated overlap-phase-time improvement;
4. Pareto frontier of critical rows vs H2D, with D2D and peer bytes annotated.

## 7. Two independent stress families

Do not pick one seed that mixes objectives.

### COMM-worst
Choose the random BR one-copy workload with the largest decode peer activation
traffic. Ties: more remote token-rank pairs, then larger critical-rank load.

### LOAD-worst
Choose the random BR one-copy workload with the largest
`sum_event max_rank_expert_rows`. Ties: peak rank rows, then peer bytes.

B128 searches:
- dataset in {MATH, ShareGPT};
- sample_seed 0..255;
- DP seed 0..255;
- BR placement seed 0..255;
using a bounded two-stage proxy then exact replay.

B256 searches:
- dataset in {MATH, ShareGPT};
- all 2048 requests;
- DP seed 0..255;
- BR placement seed 0..255.

Freeze one COMM-worst and one LOAD-worst manifest per batch. Also retain the
top-five exact candidates so the selected stress point is auditable.

Every policy is then evaluated on **both** stress families. This allows:
- CA on COMM-worst;
- LA/REP on LOAD-worst;
- cross-policy behavior on the opposite stress case.

## 8. GPU-phase-aware calibrated CPU model

Structural counters remain primary. A secondary timing model maps them onto the
planned physical execution phases:

```text
metadata exchange
rank decision
dispatch
H2D + overlapped D2D
global fetch barrier
expert compute
combine
```

### H2D calibration
Load the existing 9-MiB H2D microbenchmark artifacts. Use the median concurrent
H2D latency across the matching H100 measurements; also report a p90
sensitivity.

Per-rank H2D queues are serial, ranks execute concurrently. Because mandatory
miss counts are balanced, H2D phase lower-bound is driven by the largest rank
queue.

### D2D calibration
Until a 9-MiB H100 D2D copy microbenchmark is available, do **not** hide this
uncertainty in one guessed number. Report D2D sensitivity at:
`{0.02, 0.05, 0.10, 0.20} ms / expert copy`.

A separate model-free GPU microbenchmark is supplied to replace this grid when
four GPUs are available.

### Expert-compute calibration
Use two existing physical-profile slopes as descriptive bounds:
- GEMM-only slope;
- enclosing expert-kernel slope;

derived from the archived B8 rank-demand profile
`GPU time / decode busiest-rank expert rows`.

Also report raw critical rows because the linear extrapolation to B128/B256 is
not a physical timing claim.

### Communication calibration
Use the existing pair/transport microbenchmarks to produce fast-P2P and
conservative communication estimates. Structural dispatch/return bytes and
active rank-pair counts are always reported alongside the estimated time.

### Controller time
Do not fold Python/Numba search time into the GPU phase estimate. Controller
implementation overhead is reported separately and must be measured physically
before an E2E claim.

## 9. Prefill versus decode

Report three scopes separately:

- `prefill`: first 48 MoE layer events;
- `decode`: remaining 256*48 events;
- `moe_total`: prefill + decode.

Replica creation is evaluated in two modes on final selected configurations:
1. `decode-only`: prefill uses the base placement/cache policy; replicas start
   only in decode.
2. `all-phase`: replicas may also be created during prefill and persist into
   decode.

Interpretation:
- TPOT claims are decode-only.
- Prefill matters for TTFT/E2E, so it must be characterized if the paper claims
  end-to-end inference.
- The CPU model covers only MoE phases; attention and other prefill kernels are
  outside the modeled prefill time. Do not claim full TTFT speedup from this
  replay alone.

## 10. Required comparison tables

For each B in {128,256} and stress in {COMM,LOAD}:

| Policy | Critical rows | H2D | D2D | Peer | overlap phase estimate |
|---|---:|---:|---:|---:|---:|
| BR | | | 0 | | |
| CA | | | 0 | | |
| LA | | | 0 | | |
| BR+REP | | | | | |
| CA+REP | | | | | |
| LA+REP | | | | | |

Additionally report B256/B128 gain-ratio for:
- CA over BR;
- LA over BR;
- BR+REP over BR;
- CA+REP over CA;
- LA+REP over LA.

This directly answers whether the optimization headroom grows with batch.

## 11. Stop

CPU replay and microbenchmark-calibrated modeling only.
Commit results and stop. Do not automatically launch full-model GPU timing.
