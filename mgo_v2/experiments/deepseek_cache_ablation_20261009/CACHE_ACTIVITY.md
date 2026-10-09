# Cache activity

These are separate implementation counters, not cross-system-equivalent miss counts. main_OURS H2D and peer bytes are sums of all four ranks per complete measured batch; MoE-Infinity evictions come from its EAM cache. Values are medians of the three unfiltered target repeats with full ranges.

| B/rank | Cache | System | H2D GiB | Dispatch GiB | Return GiB | EAM evictions |
|---:|---:|---|---:|---:|---:|---:|
| 16 | 20% | main_OURS | 1424.301 [1424.301, 1424.301] | 8.990 [8.990, 8.990] | 8.951 [8.951, 8.951] | — |
| 16 | 20% | MoE-Infinity (repaired) | — | — | — | 91400.000 [91356.000, 91431.000] |
| 16 | 30% | main_OURS | 1252.018 [1252.018, 1252.018] | 8.993 [8.993, 8.993] | 8.953 [8.953, 8.953] | — |
| 16 | 30% | MoE-Infinity (repaired) | — | — | — | 80841.000 [80796.000, 80891.000] |
| 16 | 40% | main_OURS | 1078.784 [1078.784, 1078.784] | 8.998 [8.998, 8.998] | 8.959 [8.959, 8.959] | — |
| 16 | 40% | MoE-Infinity (repaired) | — | — | — | — |
| 16 | 50% | main_OURS | — | — | — | — |
| 16 | 50% | MoE-Infinity (repaired) | — | — | — | — |
