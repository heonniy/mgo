# Best observed real ShareGPT: local B16, global 64, output 256

Best observed among the preregistered screened candidates; no global-optimum or corpus-average claim.

| Candidate | R TPOT (s) | G TPOT (s) | Reduction |
|---|---:|---:|---:|
| mixed_0 | 1.490962 | 1.460771 | 2.025% |
| previous_best_control | 1.025316 | 1.018766 | 0.639% |
| family_6 | 0.916275 | 0.917147 | -0.095% |

Winner token manifest: `/data2/esjung/datasets/pcie_maxgain_expansion_20261010/B16/candidates/mixed_0.json`; SHA256 `c16bc0186483e3b4316627c2341201aba4818b31e7221296c9bcb95401c9f934`.

Winner provenance: 64 distinct dataset source rows and 64 distinct input prefixes, from 52 known original conversation families (0 rows with unknown family).

Distinct dataset split records and inputs can come from the same original conversation; they are not necessarily independent conversations. Exact family multiplicities and dataset hashes are retained in maxgain_validation.json.

Independent live greedy R/G generations may change routing, miss counts and cache trajectories; same-input repetitions within each arm must match exactly.

R/G full token agreement, matching prefixes and EOS positions are retained in maxgain_validation.json. Every request runs all 256 output steps (255 decode forwards following prefill) even if EOS occurs; any resulting routing/cache differences are part of live serving evidence, not proof of a same-trace quota-only speedup.

Nomination and single-pair screen timing selected the finalists and do not enter the final median. All candidates and screen pairs remain reported. Finalist batches were frozen before counterordered final repetitions; selecting the largest final gain still has selection bias.

Request source IDs (64): [5929, 45509, 79811, 50680, 25680, 36153, 19016, 19187, 6824, 23104, 52986, 43177, 54325, 52985, 21786, 73502, 14009, 50677, 64820, 87672, 4754, 65173, 89684, 57838, 6823, 51916, 42421, 66969, 64817, 79436, 45902, 37761, 53117, 53115, 53116, 53121, 53120, 53119, 53114, 53122, 16103, 58231, 32690, 66905, 70975, 74873, 68008, 52682, 13823, 4668, 7161, 52824, 18753, 69697, 36150, 55640, 31196, 55225, 5750, 5943, 38935, 91067, 29165, 38936]

## Winner live traffic and separate diagnostic breakdown

| Policy | Mean M | M mod 4 = 2 | H2D GiB | Group A/B H2D GiB | Reload fetches |
|---|---:|---:|---:|---:|---:|
| R-NEAR | 57.502 | 25.28% | 6186.006 | 3147.126/3038.880 | 703828 |
| G-NEAR | 56.270 | 25.07% | 6053.432 | 3026.628/3026.804 | 688741 |

The live H2D and cache totals include changed greedy trajectories; identical miss sets are not assumed.

| Policy | Metadata/PLAN | Forward | H2D | Compute | Return | Attention/router/other |
|---|---:|---:|---:|---:|---:|---:|
| R-NEAR | 0.134324 | 0.037190 | 1.115977 | 0.041464 | 0.060721 | 0.107589 |
| G-NEAR | 0.132398 | 0.036550 | 1.084769 | 0.041083 | 0.059742 | 0.106174 |

Phase values are seconds per token from separate timers-only diagnostics, including rendezvous and diagnostic assignment-copy overhead. Their endpoint sum equals diagnostic TPOT; they are excluded from unprofiled primary statistics. Differences between these separate diagnostics do not establish causal contributions to the primary R/G gap.

R/G per-arm output agreement, actual expert assignments, all-rank stage completions, nested native controller/metadata call costs and all repetitions remain inspectable in the archived validation/analysis/CSV/JSONL artifacts.
