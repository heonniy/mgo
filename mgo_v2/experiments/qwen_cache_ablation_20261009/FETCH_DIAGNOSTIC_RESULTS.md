# Qwen cache/fetch diagnostic

The separate post-generation replay passed token and final-cache parity on
all four ranks for C20 and C50. Its target output tokens and H2D byte counts
also match all three clean primary repeats at each capacity. The replay is
instrumented and does not replace the primary TPOT values.

| Metric | C20 | C50 |
|---|---:|---:|
| Decode distinct-expert MAIN hit rate | 24.83% | 63.49% |
| Decode demand misses, all ranks | 211,570 | 103,025 |
| Decode H2D, all ranks | 1,859.5 GiB | 905.5 GiB |
| 9-MiB copy service median, rank range | 0.182–0.191 ms | 0.183–0.228 ms |
| 9-MiB copy service p95, rank range | 0.220–0.233 ms | 0.228–0.382 ms |
| Largest explicit main-stream H2D wait | 0.0049 ms/token | 0 ms/token |
| Original primary TPOT median | 0.6448 s/token | 0.4819 s/token |
| Separate diagnostic job's clean target TPOT | 0.5074 s/token | 0.6821 s/token |

Increasing cache capacity **does** improve logical hit rate and halves
decode demand-copy traffic for these generated trajectories. Each transfer is
substantial: a 9-MiB expert takes about 0.18–0.23 ms of GPU copy-stream
service in these runs. The explicit main-stream wait is tiny because copies
can overlap ready expert work. It does not measure all indirect PCIe/HBM
contention or ready-first wave fragmentation, so it cannot establish that
fetch has no TPOT impact.

The clean timing difference across independent jobs is much larger than the
within-job range: C20's diagnostic job was 21% faster than its original
primary median, while C50's was 42% slower. The C50 diagnostic has a long
rank-1 expert span and larger peer metadata/return wait, consistent with a
straggler in that job. This is not proof that capacity caused the straggler:
the original C50 three-repeat job was stable and fast. Cross-capacity
autoregressive outputs also differ, so the initial 25% primary curve is not
a controlled same-route estimate of copy-removal benefit. Use the clean
post-cleanup timing recheck and, if needed, fixed-route execution before a
causal performance claim.

The explicit H2D wait and CPU/current-stream spans here come from the
instrumented replay and may be distorted by instrumentation. Per-rank detail,
copy-service distributions and validation checks are in
`FETCH_DIAGNOSTIC.json`; raw traces and token IDs stay outside Git.
