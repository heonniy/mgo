# Best observed real ShareGPT: local B64, global 256, output 64

Best observed among the preregistered screened candidates; no global-optimum or corpus-average claim.

| Candidate | R TPOT (s) | G TPOT (s) | Reduction |
|---|---:|---:|---:|
| coherent_0 | 1.657002 | 1.538487 | 7.152% |
| family_7 | 1.808352 | 1.810109 | -0.097% |
| family_2 | 1.132236 | 1.141781 | -0.843% |
| previous_best_control | 1.883592 | 1.969636 | -4.568% |

Winner token manifest: `/data2/esjung/datasets/pcie_maxgain_expansion_20261010/B64/candidates/coherent_0.json`; SHA256 `9b71d39ce103d927034fd2286fe0af2a919a004ee9cdcfd185d16b63d4d1e1a1`.

Winner provenance: 256 distinct dataset source rows and 256 distinct input prefixes, from 222 known original conversation families (0 rows with unknown family).

Distinct dataset split records and inputs can come from the same original conversation; they are not necessarily independent conversations. Exact family multiplicities and dataset hashes are retained in maxgain_validation.json.

Independent live greedy R/G generations may change routing, miss counts and cache trajectories; same-input repetitions within each arm must match exactly.

R/G full token agreement, matching prefixes and EOS positions are retained in maxgain_validation.json. Every request runs all 64 output steps (63 decode forwards following prefill) even if EOS occurs; any resulting routing/cache differences are part of live serving evidence, not proof of a same-trace quota-only speedup.

Nomination and single-pair screen timing selected the finalists and do not enter the final median. All candidates and screen pairs remain reported. Finalist batches were frozen before counterordered final repetitions; selecting the largest final gain still has selection bias.

Request source IDs (256): [5925, 48143, 84606, 19844, 37446, 12606, 13675, 16910, 9152, 11751, 36852, 32820, 50346, 32627, 55660, 11401, 7054, 81437, 51624, 56134, 2941, 31309, 83290, 30056, 26665, 5428, 79153, 71591, 40298, 86036, 58554, 12751, 62529, 84700, 40694, 29026, 50173, 26356, 59139, 1866, 90411, 51339, 15310, 26684, 79872, 51543, 91838, 25538, 68016, 73664, 86115, 86211, 229, 52257, 24299, 79995, 72146, 93440, 65222, 54378, 15866, 32874, 82966, 34092, 48300, 69420, 58, 67892, 73161, 26651, 58716, 53570, 11403, 30787, 85951, 31706, 43916, 33078, 38601, 20621, 20034, 72153, 3662, 81398, 50174, 42159, 24117, 48030, 66790, 49, 79282, 41012, 15156, 4807, 14510, 57123, 71160, 59184, 7217, 16907, 52585, 29068, 90959, 71434, 51547, 13204, 4871, 82798, 50232, 41646, 79087, 191, 84109, 4848, 85993, 83033, 36836, 78529, 81429, 31846, 192, 59635, 66788, 26649, 74, 51070, 983, 90882, 20965, 66208, 27128, 63046, 84188, 73170, 6444, 27076, 54695, 86216, 11919, 48142, 54331, 93444, 88887, 2601, 12608, 43088, 88180, 75453, 13627, 7911, 20069, 13102, 12307, 56428, 19088, 79993, 3788, 13844, 1718, 63421, 58432, 54229, 58428, 36393, 82794, 75836, 59, 34537, 7010, 81472, 26354, 89377, 59929, 75181, 52996, 7731, 555, 26288, 22158, 71348, 25368, 86474, 5519, 7055, 57998, 83204, 68540, 13087, 72144, 79997, 61468, 36864, 3684, 57748, 16199, 91475, 25885, 53577, 6204, 83787, 91901, 19778, 82325, 55378, 26310, 19766, 69132, 40188, 21746, 53573, 46047, 21071, 996, 48551, 73441, 123, 42366, 26273, 60332, 84590, 72149, 32847, 82695, 10735, 46043, 80709, 82178, 88705, 26642, 92087, 36895, 39480, 51106, 38383, 76889, 91060, 4862, 64033, 23348, 78823, 80004, 93863, 79542, 1647, 17667, 1825, 84950, 53448, 64122, 40187, 74074, 16254, 76657, 58424, 52285, 32777]

## Winner live traffic and separate diagnostic breakdown

| Policy | Mean M | M mod 4 = 2 | H2D GiB | Group A/B H2D GiB | Reload fetches |
|---|---:|---:|---:|---:|---:|
| R-NEAR | 59.627 | 25.50% | 1584.765 | 805.816/778.948 | 180310 |
| G-NEAR | 59.652 | 25.53% | 1585.433 | 792.809/792.624 | 180386 |

The live H2D and cache totals include changed greedy trajectories; identical miss sets are not assumed.

| Policy | Metadata/PLAN | Forward | H2D | Compute | Return | Attention/router/other |
|---|---:|---:|---:|---:|---:|---:|
| R-NEAR | 0.149097 | 0.040181 | 1.157957 | 0.043436 | 0.062703 | 0.108159 |
| G-NEAR | 0.145271 | 0.039229 | 1.149902 | 0.042254 | 0.060452 | 0.103927 |

Phase values are seconds per token from separate timers-only diagnostics, including rendezvous and diagnostic assignment-copy overhead. Their endpoint sum equals diagnostic TPOT; they are excluded from unprofiled primary statistics. Differences between these separate diagnostics do not establish causal contributions to the primary R/G gap.

R/G per-arm output agreement, actual expert assignments, all-rank stage completions, nested native controller/metadata call costs and all repetitions remain inspectable in the archived validation/analysis/CSV/JSONL artifacts.
