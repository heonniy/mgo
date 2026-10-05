# B128 cache-regime policy comparison

## Scope

Run only two cache settings at local batch 128:

| Setting | Main cache | Main slots (R4) |
|---|---:|---:|
| C30/B128 | 30% | 461 / 461 / 461 / 461 |
| C60/B128 | 60% | 922 / 922 / 921 / 921 |

Do not run the B8 cells in this checkpoint.

## Frozen runtime

Use **V3_OPT_PF_OVERLAP** for every cell. The stable three-arm audit selected
V3 as the timing candidate and it has the lowest absolute TPOT among the common
runtime arms. The policy experiment must not switch runtime arms to amplify a
relative policy gain.

Freeze:
- R4, physical GPUs 0,1,4,5;
- local batch 128;
- decode horizon 64 for this characterization checkpoint;
- P=2 prefetch slots per rank;
- T2 trigger;
- overlap/ready-first enabled;
- fused forward A2A and coalesced return A2A;
- BF16 common stack;
- substitution OFF;
- same eviction, NCCL, affinity, frozen requests/routes/weights/teacher tokens.

## Policies

Compare exactly four admission policies:

1. **BR** — balanced mandatory-H2D quota with random expert identity. Baseline.
2. **OLD_CA** — historical static token-rank fanout Hungarian objective
   (physical adapter id 3).
3. **FCA** — dynamic packet-aware fanout admission (id 5). Each committed miss
   updates the token->destination-rank mask so co-located experts share an
   already-required forward/return packet.
4. **LA_CA** — joint policy (id 6). Preserve the load-aware min-max compute
   decision first and use packet-aware communication as a lexicographic
   secondary objective.

Do not include standalone LA or demand-locality CA(id 1) in this checkpoint.

## Primary comparison

For each cache setting S and policy p:

```
Gain_p(S) = (TPOT_BR(S) - TPOT_p(S)) / TPOT_BR(S)
```

Primary result table:

| Cache | BR TPOT | OLD_CA gain | FCA gain | LA_CA gain |
|---|---:|---:|---:|---:|
| C30/B128 | | | | |
| C60/B128 | | | | |

Always report absolute TPOT beside relative gain.

## Mechanism attribution

Collect the same mechanism metrics for all four policies:
- exposed H2D;
- forward A2A;
- return A2A;
- expert compute;
- controller overhead;
- per-rank H2D/communication/expert imbalance;
- remote token-rank packet count;
- max rank-incident packet load.

The CA gate is physical: fewer packet metrics alone are insufficient. FCA is
useful only if the reduction reaches A2A latency and then TPOT.

The joint-policy gate is whether LA_CA reduces the dominant critical-path
component without increasing another component enough to erase the gain.

## Execution discipline

Use paired/interleaved ordering within each cache setting to reduce drift.
Apply the existing stable repeat rule. Keep C30 and C60 as separate groups and
do not pool absolute timings across them.

No performance claim is made by this planning commit.
