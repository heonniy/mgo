# Best observed real ShareGPT: local B16, global 64, output 128

Best observed among the preregistered screened candidates; no global-optimum or corpus-average claim.

| Candidate | R TPOT (s) | G TPOT (s) | Reduction |
|---|---:|---:|---:|
| previous_best_control | 1.079430 | 1.062214 | 1.595% |
| mixed_0 | 1.492919 | 1.498482 | -0.373% |
| family_5 | 1.008102 | 1.033796 | -2.549% |
| mixed_4 | 1.420470 | 1.465444 | -3.166% |

Winner token manifest: `/data2/esjung/datasets/pcie_maxgain_expansion_20261010/B16/previous_best_control.json`; SHA256 `fa0585a130c1093e796b119a71c57ffc57d8e44fcac7388862ccff3d883c830e`.

Winner provenance: 64 distinct dataset source rows and 64 distinct input prefixes, from 1 known original conversation families (0 rows with unknown family).

Distinct dataset split records and inputs can come from the same original conversation; they are not necessarily independent conversations. Exact family multiplicities and dataset hashes are retained in maxgain_validation.json.

Independent live greedy R/G generations may change routing, miss counts and cache trajectories; same-input repetitions within each arm must match exactly.

R/G full token agreement, matching prefixes and EOS positions are retained in maxgain_validation.json. Every request runs all 128 output steps (127 decode forwards following prefill) even if EOS occurs; any resulting routing/cache differences are part of live serving evidence, not proof of a same-trace quota-only speedup.

Nomination and single-pair screen timing selected the finalists and do not enter the final median. All candidates and screen pairs remain reported. Finalist batches were frozen before counterordered final repetitions; selecting the largest final gain still has selection bias.

Request source IDs (64): [26640, 26641, 26642, 26643, 26644, 26645, 26646, 26647, 26649, 26650, 26651, 26652, 26653, 26654, 26655, 26656, 26658, 26659, 26660, 26661, 26662, 26663, 26664, 26665, 26666, 26667, 26668, 26669, 26670, 26671, 26672, 26673, 26674, 26675, 26676, 26677, 26678, 26679, 26680, 26681, 26682, 26683, 26684, 26685, 26686, 26687, 26689, 26690, 26691, 26692, 26693, 26694, 26695, 26696, 26697, 26699, 26700, 26701, 26703, 26704, 26705, 26706, 26707, 26708]

## Winner live traffic and separate diagnostic breakdown

| Policy | Mean M | M mod 4 = 2 | H2D GiB | Group A/B H2D GiB | Reload fetches |
|---|---:|---:|---:|---:|---:|
| R-NEAR | 32.749 | 25.31% | 1754.640 | 904.263/850.377 | 199615 |
| G-NEAR | 32.493 | 24.21% | 1740.911 | 870.583/870.328 | 198049 |

The live H2D and cache totals include changed greedy trajectories; identical miss sets are not assumed.

| Policy | Metadata/PLAN | Forward | H2D | Compute | Return | Attention/router/other |
|---|---:|---:|---:|---:|---:|---:|
| R-NEAR | 0.137340 | 0.041835 | 0.665807 | 0.040584 | 0.059954 | 0.113918 |
| G-NEAR | 0.141693 | 0.044137 | 0.652770 | 0.042274 | 0.062200 | 0.116037 |

Phase values are seconds per token from separate timers-only diagnostics, including rendezvous and diagnostic assignment-copy overhead. Their endpoint sum equals diagnostic TPOT; they are excluded from unprofiled primary statistics. Differences between these separate diagnostics do not establish causal contributions to the primary R/G gap.

R/G per-arm output agreement, actual expert assignments, all-rank stage completions, nested native controller/metadata call costs and all repetitions remain inspectable in the archived validation/analysis/CSV/JSONL artifacts.
