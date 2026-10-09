# Best observed real ShareGPT: local B32, global 128, output 64

Best observed among the preregistered screened candidates; no global-optimum or corpus-average claim.

| Candidate | R TPOT (s) | G TPOT (s) | Reduction |
|---|---:|---:|---:|
| previous_best_control | 1.583033 | 1.528744 | 3.429% |
| coherent_0 | 1.445144 | 1.400015 | 3.123% |
| family_5 | 1.315163 | 1.317125 | -0.149% |
| family_2 | 0.872532 | 0.891824 | -2.211% |

Winner token manifest: `/data2/esjung/datasets/pcie_maxgain_expansion_20261010/B32/previous_best_control.json`; SHA256 `7b2f4caecc2985407ee6b0bc202b02d25c603e813404a990b9c410c9fd0398af`.

Winner provenance: 128 distinct dataset source rows and 128 distinct input prefixes, from 59 known original conversation families (0 rows with unknown family).

Distinct dataset split records and inputs can come from the same original conversation; they are not necessarily independent conversations. Exact family multiplicities and dataset hashes are retained in maxgain_validation.json.

Independent live greedy R/G generations may change routing, miss counts and cache trajectories; same-input repetitions within each arm must match exactly.

R/G full token agreement, matching prefixes and EOS positions are retained in maxgain_validation.json. Every request runs all 64 output steps (63 decode forwards following prefill) even if EOS occurs; any resulting routing/cache differences are part of live serving evidence, not proof of a same-trace quota-only speedup.

Nomination and single-pair screen timing selected the finalists and do not enter the final median. All candidates and screen pairs remain reported. Finalist batches were frozen before counterordered final repetitions; selecting the largest final gain still has selection bias.

Request source IDs (128): [26640, 26641, 26642, 26643, 26644, 26645, 26646, 26647, 26649, 26650, 26651, 26652, 26653, 26654, 26655, 26656, 26658, 26659, 26660, 26661, 26662, 26663, 26664, 26665, 26666, 26667, 26668, 26669, 26670, 26671, 26672, 26673, 26674, 26675, 26676, 26677, 26678, 26679, 26680, 26681, 26682, 26683, 26684, 26685, 26686, 26687, 26689, 26690, 26691, 26692, 26693, 26694, 26695, 26696, 26697, 26699, 26700, 26701, 26703, 26704, 26705, 26706, 26707, 26708, 26711, 26712, 26713, 26714, 26715, 26718, 59073, 58975, 6761, 57686, 80924, 48871, 89216, 47843, 6119, 79848, 57714, 48634, 48279, 28421, 40078, 7412, 69534, 23000, 8042, 76978, 50637, 5372, 66214, 89810, 38808, 37219, 39645, 13819, 27948, 8321, 62516, 52028, 86882, 32841, 93245, 89229, 47123, 85119, 45259, 9346, 16883, 60282, 49320, 74093, 11054, 84261, 21249, 26843, 47050, 90735, 89392, 43425, 8473, 3867, 27593, 18451, 8285, 60605]

## Winner live traffic and separate diagnostic breakdown

| Policy | Mean M | M mod 4 = 2 | H2D GiB | Group A/B H2D GiB | Reload fetches |
|---|---:|---:|---:|---:|---:|
| R-NEAR | 58.942 | 25.36% | 1566.562 | 796.597/769.966 | 178240 |
| G-NEAR | 58.612 | 24.87% | 1557.791 | 779.080/778.711 | 177242 |

The live H2D and cache totals include changed greedy trajectories; identical miss sets are not assumed.

| Policy | Metadata/PLAN | Forward | H2D | Compute | Return | Attention/router/other |
|---|---:|---:|---:|---:|---:|---:|
| R-NEAR | 0.148658 | 0.039979 | 1.148001 | 0.044560 | 0.066194 | 0.115621 |
| G-NEAR | 0.140586 | 0.037473 | 1.132293 | 0.043600 | 0.061467 | 0.106807 |

Phase values are seconds per token from separate timers-only diagnostics, including rendezvous and diagnostic assignment-copy overhead. Their endpoint sum equals diagnostic TPOT; they are excluded from unprofiled primary statistics. Differences between these separate diagnostics do not establish causal contributions to the primary R/G gap.

R/G per-arm output agreement, actual expert assignments, all-rank stage completions, nested native controller/metadata call costs and all repetitions remain inspectable in the archived validation/analysis/CSV/JSONL artifacts.
