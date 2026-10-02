# Exact-only payload capture — passed; crossover pending

The authorized one-run R4/B8 capture passed with P0 seed 42, substitution off, replication off, LRU and T0. Exactly one prefill and eight decode forwards ran, with no extra model warmup or quality evaluation. This is diagnostic capture, not a performance result.

Every raw exact expert has an observed post-admission owner. Actual send counts were recorded at the runtime all-to-all wrapper for dispatch hidden states and per-expert return outputs. They match independently computed exact-route counts; all rank-pair sends equal the corresponding receives. All 432 event plan/cache hashes match across ranks, and physical cache slots were checked on every event.

| Phase | Nonzero messages | p50 bytes | p90 bytes | p99 bytes | Max bytes |
|---|---:|---:|---:|---:|---:|
| dispatch | 4608 | 32768 | 32768 | 32768 | 32768 |
| combine | 4608 | 65536 | 90112 | 114688 | 143360 |
| union | 9216 | 32768 | 81920 | 110592 | 143360 |

Distributions include only nonzero non-self messages from the 384 decode layer events. Activation row size is 2,048 × 2 = 4,096 bytes. Metadata and protocol overhead are excluded. Separate phase distributions and all requested bucket fractions are in `payload_distribution.csv`; recorded decode count matrices are in the JSON.

The prior missing-destination audit remains at commit `0c09fae`. This newly authorized exact-only capture supersedes that blocked extraction; it does not reinterpret substituted destinations. Full raw selections, origins, exact owner maps, effective routes and plan/cache hashes remain in the rank JSON receipts under the external capture directory, with SHA-256 hashes in `exact_payload_capture_summary.json`.

Capture validation passed. The bounded five-size/two-pass crossover is the only remaining work in this follow-up; Stage 1 and replication are not authorized here.
