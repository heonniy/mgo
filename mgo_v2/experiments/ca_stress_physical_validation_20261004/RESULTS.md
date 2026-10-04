# Physical CA stress validation

Optimized communication-stress workload, not dataset-average behavior.
ShareGPT R8/local B8/cache30, Gate W128, substitution OFF, decode256,
sample81 / DP86 / BR42. Original eight-BR-seed resource reduction range:
20.7803–21.0513%, median20.9679%.

Timing stability: PASS.

| Env | Policy | n | E2E median [min,max], s | Decode median [min,max], s | TPOT median [min,max], ms |
|---|---|---:|---:|---:|---:|
| env1 | BR | 2 | 321.109 [321.090, 321.129] | 318.777 [318.737, 318.817] | 1245.223 [1245.066, 1245.380] |
| env1 | CA | 2 | 327.318 [327.064, 327.573] | 325.003 [324.760, 325.246] | 1269.538 [1268.598, 1270.479] |
| env2 | BR | 2 | 322.969 [321.821, 324.117] | 320.643 [319.512, 321.775] | 1252.515 [1248.092, 1256.938] |
| env2 | CA | 2 | 322.090 [320.961, 323.220] | 319.796 [318.671, 320.921] | 1249.199 [1244.801, 1253.598] |

| Env | E2E gain BR→CA | TPOT gain | Physical peer reduction | H2D change |
|---|---:|---:|---:|---:|
| env1 | -1.93% | -1.95% | 20.85% | 0.19% |
| env2 | 0.27% | 0.26% | 20.85% | 0.19% |

Env2 minus Env1 E2E gain: 2.206 percentage points.

Each live PLAN freezes its own physical route/token/cache trajectory. Hashes
must reproduce through COMPILE, all MEASURE repeats and COUNTERS. CPU byte
expectations are compared explicitly in counter_summary.csv; a difference is
not silently treated as an exact match. PLAN_CPU_comparison.json records
route and token differences against the original native capture.
No controller, detailed counters, profiler or compilation runs in MEASURE.
No concurrent resource scan runs between GO and process exit.
Two-sample rows report their mean (=median) and full range; three-sample rows use the median. All means and medians are retained in timing_summary.csv.
The repeat decision uses only the first two E2E and TPOT samples. Initial >5% pairs stop at two and are unstable. A third-sample outlier is also flagged descriptively, without another run.
If primary_comparison_valid is false, the displayed gain is descriptive only, not a primary BR-vs-CA result. No single-sample primary comparison is permitted.
No extra workload, policy or timing matrix is launched automatically.
