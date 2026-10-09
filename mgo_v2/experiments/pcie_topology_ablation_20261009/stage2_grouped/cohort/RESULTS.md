# Seven-policy common grouped live cohort

Independent live greedy generations on the original frozen headline64; fixed-snapshot same-miss evidence is separate.

Common grouped decode, native C++ metadata/controller and native prefill; globally serialized G2G/H2D/compute/return.

| Policy | TTFT (s) | TPOT (s) | E2E (s) | TPOT SD (s) | H2D (GiB) |
|---|---:|---:|---:|---:|---:|
| R-NEAR | 6.397757 | 1.461221 | 98.454681 | 0.017217 | 1469.479 |
| G-NEAR | 6.493899 | 1.442624 | 97.732647 | 0.007045 | 1465.523 |
| G-BR | 6.365752 | 1.453256 | 97.920875 | 0.008493 | 1470.270 |
| G-CA | 5.866776 | 1.448763 | 97.138835 | 0.006917 | 1457.016 |
| G-NUMA-CA | 5.974633 | 1.445494 | 97.035314 | 0.003203 | 1455.671 |
| R-BR | 6.565635 | 1.436730 | 97.079597 | 0.008281 | 1456.330 |
| R-CA | 6.189788 | 1.476864 | 99.198390 | 0.013100 | 1477.749 |

| Comparison | Right-arm TPOT reduction | Full token agreement |
|---|---:|---:|
| R-NEAR → G-NEAR | 1.273% | 66.41% |
| G-BR → G-CA | 0.309% | 59.96% |
| G-CA → G-NUMA-CA | 0.226% | 51.81% |
| G-NUMA-CA → G-NEAR | 0.199% | 52.83% |
| G-BR → G-NEAR | 0.732% | 62.96% |
| R-BR → G-BR | -1.150% | 61.13% |
| R-CA → G-CA | 1.903% | 58.20% |

| Policy | Metadata/PLAN | Forward | H2D | Compute | Return | Attention/router/other |
|---|---:|---:|---:|---:|---:|---:|
| R-NEAR | 0.133966 | 0.037354 | 1.077967 | 0.041679 | 0.060832 | 0.108130 |
| G-NEAR | 0.135310 | 0.037454 | 1.066083 | 0.041153 | 0.061366 | 0.108764 |
| G-BR | 0.133753 | 0.037461 | 1.068948 | 0.041275 | 0.061341 | 0.107874 |
| G-CA | 0.140043 | 0.036740 | 1.060416 | 0.041298 | 0.059689 | 0.104343 |
| G-NUMA-CA | 0.146752 | 0.038599 | 1.056431 | 0.042169 | 0.063002 | 0.111706 |
| R-BR | 0.138051 | 0.038005 | 1.064692 | 0.042304 | 0.061735 | 0.110195 |
| R-CA | 0.145794 | 0.037394 | 1.082766 | 0.042482 | 0.061683 | 0.108010 |

Phase entries are seconds/token from one separate diagnostic per arm. Their endpoint partition sums to diagnostic TPOT and includes global waiting. It is distinct from primary medians and does not establish a causal attribution of primary gaps.

| Policy | Metadata CPU ms/layer | Controller CPU ms/layer | PLAN residual CPU ms/layer |
|---|---:|---:|---:|
| R-NEAR | 0.7660 | 0.1615 | 0.8113 |
| G-NEAR | 0.7737 | 0.1614 | 0.8206 |
| G-BR | 0.7681 | 0.1547 | 0.8174 |
| G-CA | 0.6878 | 0.3514 | 0.7898 |
| G-NUMA-CA | 0.8405 | 0.3500 | 0.8481 |
| R-BR | 0.7993 | 0.1597 | 0.8286 |
| R-CA | 0.7592 | 0.3692 | 0.8290 |

All primary repetitions are retained. Diagnostics and warmups do not enter these statistics.

Post-generation RSS per rank includes shared source mappings; summing rank RSS double-counts physical source pages.

Actual assigned experts and quota rows are joined only after diagnostic tokens/cache/full traces match all primaries. Nested CPU spans are not summed as separate phases; diagnostic frontier partitions cover the complete diagnostic TPOT.

Independent greedy trajectories can change miss sets, cache states and quota values. Same-miss placement evidence is in stage2_fixed_miss; this table measures live generation.

MoE-Infinity, DeepSpeed and llama.cpp results are not included in this OURS-only table. Their required new-host measurements remain pending.
