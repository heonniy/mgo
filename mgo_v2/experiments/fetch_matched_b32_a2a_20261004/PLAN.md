# PLAN — fetch-matched B32 + routing-weight A2A removal

Date: 2026-10-04
Status: owner-authorized, queued after current physical R4 work completes.

## 1. Frozen high-level setting

- dataset: ShareGPT V3 cleaned
- existing 2,048-request exact-model candidate pool
- decode horizon: 256
- local batch: **32**
- cache ratio: **30% global**
- eviction: Gate, W=128
- substitution: **OFF**
- policies: BR and CA
- rank counts: R=4 and R=8
- two independently selected stress cells per rank count: **Peer-best** and **Critical-best** (deduplicate if identical)

Global request counts:
- R4: 128 requests
- R8: 256 requests

No new model/data capture is required unless the existing 2,048-request pool
fails validation.

## 2. Stage A — fetch-matched workload search

Search independently for R4/B32/cache30 and R8/B32/cache30.

Search variables:
- sample_seed: 0..511 (**512 samples**)
- dp_seed: 0..63 (**64 DP partitions per sample**)
- BR seed: [7,19,42,73,99,131,181,251]

The DP split must contain exactly B=32 requests per rank.

Per R, run 512 x 64 = **32,768 routing-only prescreen candidates**. Maintain
two independent rankings:
- Peer ranking: top 64 by aggregate peer-byte reduction.
- Critical ranking: top 64 by per-event critical-rank communication reduction.

Deduplicate the two top-64 lists, so at most 128 sample/DP candidates per R
enter exact Gate-cache replay. For each retained candidate, test the eight BR
seeds and deterministic CA. This is at most 1,152 exact replays per R, 2,304
across R4+R8.

### Hard validity conditions

A candidate BR/CA pair is eligible only if, over the full frozen 256-step
trace:

```
total_fetches_BR == total_fetches_CA
H2D_bytes_BR     == H2D_bytes_CA
quota_max_minus_min <= 1 for both policies
```

Because every exact expert fetch is the same 9-MiB payload, exact fetch equality
must imply exact H2D-byte equality; validate both explicitly.

Record first-fetch and reload counts separately. Prefer candidates with exact
reload equality, but total fetch/H2D equality is the hard owner requirement.

### Two independent communication objectives

For every eligible pair compute both:

1. **Aggregate peer bytes**
   ```
   Peer = sum over all remote activation/return traffic
   ```

2. **Per-event critical-rank communication**
   For every dispatch/return collective event i:
   ```
   Critical_i = max_r(send_bytes[i,r] + recv_bytes[i,r])
   Critical_total = sum_i Critical_i
   ```
   This intentionally does not collapse the whole run into one max-rank total;
   the bottleneck rank may change across layers/steps.

Select two winners independently for each R:

- **Peer-best**: among exact fetch/H2D-matched candidates, maximize absolute
  BR->CA aggregate peer-byte reduction; tie-break by relative peer reduction,
  then critical reduction, then exact reload equality.
- **Critical-best**: among exact fetch/H2D-matched candidates, maximize absolute
  BR->CA `Critical_total` reduction; tie-break by relative critical reduction,
  then aggregate peer reduction, then exact reload equality.

Do not force the two objectives to choose the same workload. If the exact same
(sample_seed, dp_seed, BR_seed) wins both, deduplicate it in physical timing.

Report both winners, their cross-metrics, the top-10 eligible alternatives for
each objective, and how many searched pairs satisfy exact fetch matching.

## 3. Common physical workload requirement

The previous live autoregressive R8 experiment showed that a physical PLAN can
diverge numerically from the original native capture, which changes route and
fetch trajectories. That is unacceptable for a strict fetch-matched mechanism
test.

Therefore the physical comparison in this packet uses a **common frozen-route,
teacher-forced decode replay** for each selected R cell:

- identical request IDs and rank membership for BR and CA;
- identical 256 generated-token sequence from the selected exact trace;
- identical per-layer top-k expert IDs and routing weights;
- the model still executes real dense layers, real expert kernels, real CPU->GPU
  expert H2D, real bounded GPU cache, and real NCCL communication;
- only expert placement differs between BR and CA.

The router computation may still execute for realistic compute cost, but its
live top-k is not allowed to change the frozen system workload.

Label these results **fixed-route physical decode replay**, not autoregressive
quality measurements.

Before timing, verify that the physical BR/CA COUNTERS exactly retain the
selected CPU search's fetch/H2D equality.

## 4. Runtime variants

### A3 — current baseline, 3 All-to-All calls per MoE layer

```
activation dispatch A2A
routing-weight A2A
expert compute with remote weight multiply
expert-output return A2A
local combine
```

### A2 — remove routing-weight A2A

```
activation dispatch A2A
expert compute WITHOUT weight multiply
expert-output return A2A
source-side routing-weight multiply
local combine
```

Mathematically:
```
return(w * E(x))  ==  w * return(E(x))
```
because the return transport is a copy.

For the source-side multiply, extend the frozen layout so every returned expert
partial is associated with its original source token and expert ID. Use the
source's frozen routing weight immediately before `index_add_`.

Do not transmit a routing-weight tensor in A2.

Expected collective count:
- all 257 forwards: A3 = 3 * 12,336 = 37,008 A2A calls/rank;
- all 257 forwards: A2 = 2 * 12,336 = 24,672 A2A calls/rank;
- reduction = 12,336 calls/rank (33.3%).

For decode-only 256 steps:
- A3 = 36,864 calls/rank;
- A2 = 24,576 calls/rank.

## 5. A2 correctness gate

Before any A2 timing:
- same frozen routes/weights/actions/cache trajectory;
- same number and byte size of activation and return packets;
- zero routing-weight A2A calls in A2;
- exact expert fetch/H2D equality between A3 and A2 for the same policy;
- compare weighted expert contributions from A3 and A2 on an untimed prefix;
- require exact equality if achievable; otherwise report max-abs/max-rel BF16
  difference and require identical final teacher-forced token/logit-argmax hash
  over a full untimed 256-step pass.

If this gate fails, stop A2 timing.

## 6. Fast physical pivot screen

Primary environment: **Env 2 only** (P2P disabled, validated SHM).

Run **R4 first, then R8** so the owner can pivot early.

Physical GPU sets:
- R4: GPUs 0,1,4,6
- R8: GPUs 0,1,2,3,4,5,6,7

Do not search or change the GPU subset after timing begins.

For each R, time both selected workloads:
- Peer-best
- Critical-best

If both objectives selected the exact same workload/BR seed, run it once.

For every unique workload, the single-shot screening matrix is:
- BR/A3 once
- CA/A3 once
- BR/A2 once
- CA/A2 once

Thus there are at most four unique workload sets and **16 initial timed runs**
(8 for R4 + 8 for R8). The minimum is 8 if Peer-best and Critical-best coincide
for both R values.

### Warmup policy

Do **not** run a full 256-step warmup before every MEASURE.

- Populate/validate compiler artifacts once per (R, runtime A3/A2) outside the
  scientific timer.
- Before the first timed use of each (R, runtime), allow only a short untimed
  8-16 decode-step readiness warmup.
- Subsequent BR/CA measurements reuse the validated compile artifacts and do
  not repeat a full schedule warmup.
- Keep `error_on_recompile=True` during MEASURE. If a timed run recompiles,
  invalidate that run rather than silently including compilation.

### Selective confirmation rule

This is an exploratory pivot screen, not the final paper timing protocol.

After the first BR/CA samples for a fixed (R, workload, runtime):
- if positive CA gain is **<1%** in both E2E/decode-wall and TPOT: no repeat;
- if positive CA gain is **>=1%** in any primary timing metric: run **exactly
  one additional confirmation pair** — BR once and CA once;
- do not add a second confirmation repeat, even if the two observations differ.

Therefore an interesting pair has two observations per policy total; an
uninteresting pair remains single-shot. Report raw values, not confidence
intervals. Final publication-quality repetition can be done later only for
surviving cells.

## 7. Timing hygiene

Reuse the validated stable harness:
- BOUNDARY-only safety monitoring;
- no concurrent PSS/smaps/nvidia-smi/ps inside timed region;
- fixed disjoint CPU affinity;
- NORMAL expert residency;
- PLAN/controller/Hungarian outside MEASURE;
- compile/warmup outside the scientific timer;
- no profiler, detailed counters or per-event prints in MEASURE;
- no recompilation in MEASURE.

COUNTERS is a separate pass.

## 8. Required comparisons

For each R:

### Placement effect with current runtime
```
gain_CA_A3 = (T_BR_A3 - T_CA_A3) / T_BR_A3
```

### Placement effect after removing the weight collective
```
gain_CA_A2 = (T_BR_A2 - T_CA_A2) / T_BR_A2
```

### Runtime optimization itself
```
gain_A2_BR = (T_BR_A3 - T_BR_A2) / T_BR_A3
gain_A2_CA = (T_CA_A3 - T_CA_A2) / T_CA_A3
```

Report E2E/decode wall/TPOT separately.

Mechanism counters next to timing:
- total fetches and H2D bytes: BR == CA by construction;
- first fetches / reloads;
- aggregate peer bytes;
- **per-event critical-rank send+recv bytes and Critical_total**;
- max per-rank whole-run send+recv bytes (diagnostic only);
- per-rank send and recv distributions;
- peer fan-out;
- activation A2A bytes;
- routing-weight A2A bytes/calls;
- return A2A bytes;
- collective call count;
- exact global/local hit rates;
- evictions.

## 9. Interpretation

This is a mechanism-isolation study.

A positive result would support:
- larger B makes payload cost more visible;
- CA reduces critical-rank communication under fixed H2D work;
- removing one small-payload collective reduces fixed synchronization overhead,
  allowing placement byte savings to translate more directly into latency.

A negative result would indicate that communication volume/collective count is
still not the dominant critical path and should trigger a deeper layer-level
breakdown rather than more sample picking.

Do not call optimized stress cells dataset-average behavior.

## 10. Stop

Commit:
- search winners/top-10;
- exact fetch-match validation;
- A2 correctness validation;
- physical timing/counters.

Then stop for owner review. No Env1 rerun, other batch/cache ratio, substitution,
replication or extra GPU-set sweep follows automatically.
