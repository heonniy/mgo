# Best observed real ShareGPT batch64

Best observed among the preregistered screened candidates; no global-optimum or corpus-average claim.

| Candidate | R TPOT (s) | G TPOT (s) | Reduction |
|---|---:|---:|---:|
| family_3 | 1.058425 | 1.008296 | 4.736% |
| mixed_0 | 1.371181 | 1.313678 | 4.194% |
| family_0 | 1.177071 | 1.154249 | 1.939% |

Winner token manifest: `/data2/esjung/mgo-results/pcie_topology_ablation_20261009/maxgain64_search_attempt1/candidates_repaired/family_3.json`; SHA256 `32431c31694c633583fcc20cc9bab56b8db5c8b271d8d0ba7856c3b4735b7506`.

Winner provenance: 64 distinct dataset source rows and 64 distinct input prefixes, from 1 known original conversation families (0 rows with unknown family).

Distinct dataset split records and inputs can come from the same original conversation; they are not necessarily independent conversations. Exact family multiplicities and dataset hashes are retained in maxgain_validation.json.

Independent live greedy R/G generations may change routing, miss counts and cache trajectories; same-input repetitions within each arm must match exactly.

R/G full token agreement, matching prefixes and EOS positions are retained in maxgain_validation.json. Every request runs all 64 output steps even if EOS occurs; any resulting routing/cache differences are part of live serving evidence, not proof of a same-trace quota-only speedup.

Nomination and single-pair screen timing selected the finalists and do not enter the final median. All candidates and screen pairs remain reported. Finalist batches were frozen before counterordered final repetitions; selecting the largest final gain still has selection bias.

Request source IDs (64): [26640, 26641, 26642, 26643, 26644, 26645, 26646, 26647, 26649, 26650, 26651, 26652, 26653, 26654, 26655, 26656, 26658, 26659, 26660, 26661, 26662, 26663, 26664, 26665, 26666, 26667, 26668, 26669, 26670, 26671, 26672, 26673, 26674, 26675, 26676, 26677, 26678, 26679, 26680, 26681, 26682, 26683, 26684, 26685, 26686, 26687, 26689, 26690, 26691, 26692, 26693, 26694, 26695, 26696, 26697, 26699, 26700, 26701, 26703, 26704, 26705, 26706, 26707, 26708]

## Winner live traffic and separate diagnostic breakdown

| Policy | Mean M | M mod 4 = 2 | H2D GiB | Group A/B H2D GiB | Reload fetches |
|---|---:|---:|---:|---:|---:|
| R-NEAR | 31.346 | 24.67% | 833.124 | 429.794/403.330 | 94780 |
| G-NEAR | 30.677 | 23.74% | 815.344 | 407.681/407.663 | 92759 |

The live H2D and cache totals include changed greedy trajectories; identical miss sets are not assumed.

| Policy | Metadata/PLAN | Forward | H2D | Compute | Return | Attention/router/other |
|---|---:|---:|---:|---:|---:|---:|
| R-NEAR | 0.141520 | 0.042917 | 0.641334 | 0.041506 | 0.061563 | 0.121095 |
| G-NEAR | 0.132548 | 0.040932 | 0.613247 | 0.039574 | 0.057080 | 0.111418 |

Phase values are seconds per token from separate timers-only diagnostics, including rendezvous and diagnostic assignment-copy overhead. Their endpoint sum equals diagnostic TPOT; they are excluded from unprofiled primary statistics. Differences between these separate diagnostics do not establish causal contributions to the primary R/G gap.

R/G per-arm output agreement, actual expert assignments, all-rank stage completions, nested native controller/metadata call costs and all repetitions remain inspectable in the archived validation/analysis/CSV/JSONL artifacts.

Owner-requested Git pushes ran concurrently with part of the final measurements. The cohort-local ADMIN_GIT_PUSH*.json receipts record observation timestamps, before/after phase labels and verified remote SHAs. These observations approximately bracket administration, not exact pack/upload duration. Every timing is retained and the originally prescribed three/five-repeat stability rule is unchanged.
