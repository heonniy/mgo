# CPU-only replica Pareto screen

**GO_FOR_OWNER_REVIEW** on the frozen eight-decode trace. All five budgets and 2,160 layer events passed validation.

There are 5 nondominated rho settings: [0.0, 0.125, 0.25, 0.5, 0.75]. F is rho=0; C is rho=0.75; K is 0.25. Moving F to C increases decode expert H2D by **160.20%** and reduces decode peer activation bytes by **100.00%**. The decision is a trace-screen result; physical validation has not started.

## Primary decode plane

GiB and MiB are binary units. H2D includes every physical first copy, reload and replica; peer bytes include only non-self dispatch plus combine activation rows.

| rho | Duplicate cap | Expert H2D GiB | Peer MiB | Remote token-rank pairs | Nondominated |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0 | 154.9951 | 319.4531 | 23,179 | yes |
| 0.125 | 230 | 183.3574 | 250.3398 | 18,342 | yes |
| 0.25 | 460 | 219.0762 | 185.2969 | 14,376 | yes |
| 0.5 | 921 | 340.5498 | 47.9570 | 4,020 | yes |
| 0.75 | 1382 | 403.2949 | 0.0000 | 0 | yes |

## Decode fetches and residency

| rho | First-ever copies | Reloads | Replicas | Total fetches |
|---:|---:|---:|---:|---:|
| 0 | 194 | 17,441 | 0 | 17,635 |
| 0.125 | 194 | 17,850 | 2,818 | 20,862 |
| 0.25 | 194 | 18,157 | 6,575 | 24,926 |
| 0.5 | 194 | 18,388 | 20,165 | 38,747 |
| 0.75 | 194 | 18,388 | 27,304 | 45,886 |

| rho | Mean / peak duplicate fraction | Mean unique experts | Local exact hits | Resident remote services | Final local services |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 / 0.0000 | 1843.00 | 1.79% | 4.91% | 40.39% |
| 0.125 | 0.1248 / 0.1248 | 1613.00 | 1.38% | 2.55% | 53.47% |
| 0.25 | 0.2496 / 0.2496 | 1383.00 | 0.64% | 0.67% | 66.37% |
| 0.5 | 0.4994 / 0.4997 | 922.61 | 0.00% | 0.00% | 91.60% |
| 0.75 | 0.5806 / 0.7238 | 772.88 | 0.00% | 0.00% | 100.00% |

Hit fractions use residency at event entrance, whereas final local service also includes copies fetched for this event. Resident remote service requires that the actual remote serving copy existed at entrance. Denominators are raw expert routes. Occupancy means average post-event state; peaks include intermediate state.

## Prefill and full-trace context

| rho | Prefill H2D GiB | Prefill peer MiB | Full H2D GiB | Full peer MiB |
|---:|---:|---:|---:|---:|
| 0 | 46.3008 | 3410.8828 | 201.2959 | 3730.3359 |
| 0.125 | 54.2461 | 2825.9805 | 237.6035 | 3076.3203 |
| 0.25 | 64.2920 | 2217.9102 | 283.3682 | 2403.2070 |
| 0.5 | 97.7871 | 1096.5156 | 438.3369 | 1144.4727 |
| 0.75 | 169.5938 | 0.0000 | 572.8887 | 0.0000 |

Each budget replays the same 48 prefill + 384 decode events from an empty cache, with 1,843 slots split [461,461,461,460]. Prefill initializes state and is excluded from Pareto selection. Total routes are 928,896 in prefill and 98,304 in decode. The trace contains 5,462 unique (layer,expert) keys, all fetched first-ever once per budget.

## Validation and reproducibility

- Seven CPU policy tests passed, including hand-counted traffic, exact greedy savings, incremental dispatch effects, tie breaks, LRU, primary promotion, inactive unique-copy eviction, active-copy protection and atomic mandatory-admission failure.
- Synthetic differential coverage: 48 independent rho=0 events; 192 replica events checked against full candidate traffic recomputation and a deterministic twin replay.
- Actual rho=0: all 432 events match a separately written slot-array implementation in full state, route destinations, dispatch/combine matrices and fetch classes. Zero duplicates at every event.
- Every actual event checks exact resident destinations, physical capacities, duplicate cap, primary validity, slot ownership and send/receive transposes. All CSV sums and full=prefill+decode identities were cross-checked against JSON; Pareto dominance was independently recomputed.
- All four raw capture hashes match the prior validated receipts. The raw routes are frozen; captured P0 placements are intentionally not replay targets.
- One CPU process, one numerical-library thread, hard address-space limit 2 GiB; measured peak RSS **201.88 MiB**, elapsed **98.22 seconds**. No Torch import, GPU run, model generation, NCCL call, or new quality/performance measurement.
- Policy/protocol commit: `98a3a6505c41dad39e0f3fa954062b13550e0cd8`. Capture result commit: `ed7f82b6a323460c18374c45f24018bc48a899f4`.

See [frozen conventions and commands](REPLICA_REPLAY_PROTOCOL.md), [summary CSV](replica_pareto_screen.csv), [event CSV](replica_pareto_events.csv), [full JSON](replica_pareto_screen.json), and [validation and output hashes](replica_pareto_validation.json).

## Interpretation and stop boundary

These are counterfactual byte counts under one common deterministic first-copy/LRU policy and a greedy current-byte replica rule. From F to C, decode reloads increase from 17,441 to 18,388 and replica fetches from zero to 27,304. Mean unique coverage falls from 1,843 to 772.88 experts. At C, every route executes locally after current-event fetches, while entrance local hits are zero: the eliminated peer traffic comes with substantial repeated CPU loading. The rho=.75 cap is not reached because all current demand is already local before the cap fills.

This short frozen trace does not establish runtime speedup, steady-state behavior, numerical quality, or an optimal placement policy. F/C are selected only on decode totals; K, when present, is the greatest positive inward distance from the normalized endpoint chord. Endpoint percentages use F as denominator.

The three gate conditions pass. A longer trace or physical F/K/C validation may be considered by the owner; this task ends at the CPU result.
