# Stage B2 runtime gap attribution

Mechanism gate: **True**. Placement causality: **not established**. Stage C remains blocked; no revised delta model or oracle executed.

Four primary Env1 R4/B128 C30/C60 × BR/FCA captures, first8decode steps. All capture runs must pass unchanged token/cache/copy/controller checks. No samples dropped. GPU2/3/6/7 belong to another user and are excluded from our jobs/idle restoration. One initial FCA capture was aborted by an overly broad occupancy guard; its failure is preserved. The original pre-arrival BR capture is supplemental only; the four primary captures use the recovered shared-host state. Shared workload is observed, not assumed stationary.

## Collective residency accounting

| Cache | Policy | Phase | Residency ms/step | Base | Measured start-skew wait | Signed residual | Inflation explained |
|---|---|---|---:|---:|---:|---:|---:|
| C30 | BR | forward | 16.149 | 1.509 | 15.301 | -0.661 | 95.5% |
| C30 | BR | return | 92.449 | 1.489 | 91.540 | -0.580 | 99.4% |
| C30 | FCA | forward | 41.211 | 1.363 | 40.985 | -1.136 | 97.2% |
| C30 | FCA | return | 245.700 | 1.402 | 244.944 | -0.647 | 99.7% |
| C60 | FCA | forward | 35.797 | 1.390 | 35.499 | -1.092 | 96.8% |
| C60 | FCA | return | 251.657 | 1.419 | 250.822 | -0.584 | 99.8% |
| C60 | BR | forward | 17.905 | 1.513 | 17.059 | -0.667 | 96.0% |
| C60 | BR | return | 99.970 | 1.491 | 99.060 | -0.581 | 99.4% |

Values are mean-rank ms per step, not additive TPOT. Explained fraction uses absolute per-event residual to prevent cancellation. Start-skew wait is max(rank GPU NCCL start) minus that rank start; it is not summed again with host skew/ready skew. Residual remains signed and all samples remain.

## Expert loop accounting

| Cache | Policy | Tau | Actual compiled GPU | Wrapper GPU | Host loop | Host ready wait | Host compiled call | Host other | GPU error | Host error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C30 | BR | 20.320 | 17.970 | 8.648 | 989.748 | 0.528 | 606.708 | 69.222 | 8.8% | 59.2% |
| C30 | FCA | 20.320 | 17.934 | 8.616 | 987.206 | 0.084 | 607.887 | 67.820 | 9.0% | 59.5% |
| C60 | FCA | 20.320 | 18.048 | 8.615 | 714.280 | 0.017 | 449.388 | 48.140 | 8.5% | 60.1% |
| C60 | BR | 20.320 | 18.255 | 8.645 | 702.087 | 0.000 | 444.063 | 46.717 | 7.7% | 60.4% |

## Nonoverlapping expert-loop timeline

| Cache | Policy | Full span | Compiled GPU | Ready wait exclusive | Wrapper exclusive | Other host exclusive | Unexplained | Tau-based error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C30 | BR | 989.748 | 17.970 | 0.528 | 902.283 | 68.968 | 0.000 | 0.2% |
| C30 | FCA | 987.206 | 17.934 | 0.084 | 901.583 | 67.605 | 0.000 | 0.2% |
| C60 | FCA | 714.280 | 18.048 | 0.017 | 648.093 | 48.122 | 0.000 | 0.3% |
| C60 | BR | 702.087 | 18.255 | 0.000 | 637.779 | 46.053 | 0.000 | 0.3% |

This uses a physical timeline union, assigning GPU compiled work first, then nonoverlapping host ready wait, then nonoverlapping wrapper CPU/GPU work, then other host-loop time. The last category is measured but not causally identified. Tau replaces only compiled GPU service for the25% accounting gate. The separate host-substitution error above exposes launch overhead and is not substituted for this physical-loop gate.

Slot DMA completion is matched by expert key and submission order. A GPU wait upper bound uses prior same-stream work, host wait entry, DMA completion and the next gather kernel; it is not asserted to be an isolated stall measurement. Values remain in EXPERT_ATTRIBUTION.

Host and GPU figures are different clocks/interval domains and overlap. GPU estimate = tau + measured gather/weight GPU service. Host budget estimate = tau + measured non-compiled host budget; its error exposes kernel enqueue/backpressure beyond service time. Exact host phase partition is an accounting identity, not proof of causality. Ready-select includes ready-wait; the two are not double counted.

## Interpretation

A near-one-for-one link between arrival skew and NCCL residency can explain where kernels wait without explaining why ranks arrive late. Expert ownership is exact and service time is measurable, but host enqueue, ready-slot submission, scheduling and staging remain mixed. These terms cannot be labeled placement-controllable solely because their BR/FCA averages differ. No global fitted scale or arbitrary common-offset subtraction is used.

Per-term BR→FCA deltas, residual policy-invariance checks, phase timelines and clock-calibration uncertainty are in the JSON/CSV artifacts. Any still-unidentified timing remains explicit, rather than assigned to wire service or expert service.

## Stop

Stop for owner review per STAGE_B2_RUNTIME_GAP_ATTRIBUTION.md. This diagnostic does not authorize a new controller or oracle.

- No timing exclusions or latency-selected tail events; prefix is fixed8steps.
- B2 instrumentation changes host scheduling; no primary TPOT claim.
- Host ready_wait measures submission blocking and wait insertion, not a direct isolated GPU DMA-stall duration.
- Expert host budget and GPU-active accounting are distinct; adding them is forbidden.
- Class-A dominance remains unproven; successful residency accounting is not sufficient for an oracle.

## Host-work predictability probe

The number/shape of expert host calls is owner-dependent, so it is an admission-controllable work-volume candidate. This is stronger than labeling all host work uncontrollable, but weaker than validating a placement cost. No compilation occurred inside captures: the long compiled-host-call range is invocation/enqueue/backpressure of an already compiled function, not compiler time.

| Cache | Policy | Held-out rank events | Host-loop Spearman | Aggregate error |
|---|---|---:|---:|---:|
| C30 | BR | 768 | 0.728 | 0.4% |
| C30 | FCA | 768 | 0.839 | 0.0% |
| C60 | BR | 768 | 0.713 | 18.6% |
| C60 | FCA | 768 | 0.863 | 19.4% |

Calibration uses only BR steps0–3 and direct per-call measured host medians plus measured noncompiled overhead per expert, separately per cache. BR/FCA steps4–7 are held out; no old TPOT coefficients or arbitrary global scale are fitted. The C30 held-out return-wait delta error is62.3%; C60 is23.8%. This prevents a robust placement-causality pass across both caches.

| Cache | Policy | Predicted delta ms/step | Observed profile delta ms/step | Delta Spearman | Delta error |
|---|---|---:|---:|---:|---:|
| C30 | OLD_CA | 165.798 | 100.394 | 0.428 | 65.1% |
| C30 | FCA | 335.837 | 194.294 | 0.644 | 72.8% |
| C30 | LA_CA | -28.186 | 15.452 | 0.325 | 282.4% |
| C60 | OLD_CA | 62.847 | 42.360 | 0.644 | 48.4% |
| C60 | FCA | 222.290 | 155.769 | 0.700 | 42.7% |
| C60 | LA_CA | -14.733 | -16.921 | 0.478 | 12.9% |

These old-capture comparisons are exploratory diagnostic rejection checks, not an accepted revised oracle model. Targets are instrumented causal event spans, not primary TPOT. They fail the25% delta-error gate and some correlation/direction checks; no revised model is approved.

Host collective-entry spread and GPU NCCL-start spread correlate0.980–0.994 forward and0.9994–0.9998 return. This supports host-arrival/launch imbalance as the main measured origin of NCCL waiting. It does not identify a unique Python/GIL, driver or OS cause. Slot GPU-wait upper bounds are0–0.534 mean-rank ms/step in these captures; do not equate negligible exposed slot wait with zero CPU staging interference.

The practical finding is that wire service and expert GPU arithmetic omit the dominant host-side runtime budget. The next cost model would need a validated owner-dependent host-work component and its regime dependence. No runtime/controller optimization, new policy, new LA_CA capture or oracle was launched.
