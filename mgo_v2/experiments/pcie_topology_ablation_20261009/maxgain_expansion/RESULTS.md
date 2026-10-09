# Best-input batch and output-length expansion

Local batch is per GPU/rank; global requests are four times local batch. Output includes the first prefill-produced token, so decode forwards = output tokens minus one. EOS never stops generation.

The previous winner is a prespecified control. B16 keeps its exact64 ordered inputs; B32/B64 add distinct whole-corpus requests and repartition across ranks. Newly searched winners can differ by cell and must not be treated as matched-input scaling.

All retained primary repeats determine medians. The short-routing proxy and single-pair screening scores are selection evidence only. Finalists are frozen before repeats. Positive gain is not assumed; separate diagnostics and token agreement expose changed live trajectories.

| Local/global batch | Output | State | Control R / G TPOT (s) | Control gain | Searched best | Best gain |
|---|---:|---|---:|---:|---|---:|
| 16/64 | 64 | HISTORICAL_PASS | 1.058425 / 1.008296 | 4.736% | family_3 | 4.736% |
| 16/64 | 128 | PENDING | — | — | — | — |
| 16/64 | 256 | PENDING | — | — | — | — |
| 32/128 | 64 | PASS | 1.583033 / 1.528744 | 3.429% | previous_best_control | 3.429% |
| 32/128 | 128 | PENDING | — | — | — | — |
| 32/128 | 256 | PENDING | — | — | — | — |
| 64/256 | 64 | PENDING | — | — | — | — |
| 64/256 | 128 | PENDING | — | — | — | — |
| 64/256 | 256 | PENDING | — | — | — | — |

Historical B16/O64 retained the disclosed concurrent CPU/Git administration. New primary windows perform no builds, analytical generation or Git packing. GPU lease/burn transitions and NUMA-shared pinned108GiB source proofs are archived per job.

Each completed cell retains exact source IDs/manifests hashes, cold-cache/full-trace parity, assigned experts, all-rank serialized phase validation, memory, every primary and separate TPOT breakdown. Pending cells have no result claim.
