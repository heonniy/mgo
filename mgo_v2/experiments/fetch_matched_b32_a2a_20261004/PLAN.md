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
- one selected stress cell per rank count

Global request counts:
- R4: 128 requests
- R8: 256 requests

No new model/data capture is required unless the existing 2,048-request pool
fails validation.

## 2. Stage A — fetch-matched workload search

Search independently for R4/B32/cache30 and R8/B32/cache30.

Search variables:
- sample_seed: 0..511
- dp_seed: 0..511
- BR seed: [7,19,42,73,99,131,181,251]

The DP split must contain exactly B=32 requests per rank.

Use a routing-only prescreen to retain the best 128 sample/DP candidates per R,
then run exact Gate-cache replay for every retained candidate and BR seed.

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

### Communication score

For every eligible pair compute both:
- aggregate peer bytes;
- per-rank send+receive bytes and the maximum critical-rank communication load.

Selection order for each R:
1. require CA not to increase max-rank communication;
2. maximize reduction in max-rank communication;
3. tie-break by absolute aggregate peer-byte reduction;
4. tie-break by relative aggregate peer reduction;
5. tie-break by exact reload equality.

This keeps the selected stress example aligned with collective critical-path
latency rather than aggregate bytes alone.

Report the chosen sample_seed, dp_seed and BR seed plus the top-10 eligible
alternatives. Also report how many searched pairs satisfy exact fetch matching.

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

## 6. Physical timing scope

Primary environment: **Env 2 only** (P2P disabled, validated SHM).

Reason: this packet tests whether larger messages plus one fewer collective
expose the placement benefit under the project's expensive-communication
condition. The earlier R8/B8 packet already provides Env1/Env2 context.

Physical GPU sets:
- R8: GPUs 0,1,2,3,4,5,6,7
- R4: GPUs 0,1,4,6 (the predeclared representative quartet from the prior
  pair-matrix discussion)

Do not search or change the GPU subset after timing begins.

Matrix:
- R8-best: BR/A3, CA/A3, BR/A2, CA/A2
- R4-best: BR/A3, CA/A3, BR/A2, CA/A2

Start with two clean MEASURE repeats per combination.

Repeat rule:
```
relative_diff = abs(T1-T2) / mean(T1,T2)
```
- <=2% for both E2E/decode-wall and TPOT: stop at 2;
- >2% and <=5%: add exactly one third repeat;
- >5%: mark unstable and stop that comparison.

No single-sample primary result.

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
- max per-rank send+recv bytes;
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
