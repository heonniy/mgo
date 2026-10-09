# Output token stability

Full 64-token request agreement across the three target repeats (out of 64 requests). Raw IDs remain outside Git.

| Cache | 1 versus 2 | 1 versus 3 | 2 versus 3 |
|---:|---:|---:|---:|
| 20% | 64/64 | 64/64 | 64/64 |
| 30% | 64/64 | 64/64 | 64/64 |
| 40% | 64/64 | 64/64 | 64/64 |
| 50% | — | — | — |

Across capacities, identical input requests can lead to different autoregressive output paths. Compare the second target repeat in each passed cell; these are aggregate counts, not raw token IDs.

| Cache pair | Same first token | Same complete 64-token sequence | Same token positions |
|---|---:|---:|---:|
| C20–C30 | 64/64 | 23/64 | 2735/4096 |
| C20–C40 | 64/64 | 31/64 | 3001/4096 |
| C30–C40 | 64/64 | 26/64 | 2792/4096 |
