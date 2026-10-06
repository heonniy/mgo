# Static layer-split small cell

All3 primary repeats pass manifest coverage,64-token output, native-clock identities and static-budget configuration checks. Timing is UNSTABLE; no samples removed.

- TTFT: median 37.411425500s; full range [35.822114859, 41.396564603]s; spread 14.589%.
- TPOT: median 0.255360333s; full range [0.251160254, 0.267041571]s; spread 6.159%.
- E2E: median 54.235044500s; full range [51.909815859, 57.219660603]s; spread 9.751%.

One bounded follow-up is needed before headline selection. Native BF16 weights, FP16 KV;14 expert layers remain GPU resident,34 execute on CPU with host-op offload disabled. Actual CUDA KV allocations were verified in static_smoke1.

One confirmation job is queued after CLEANUP_RECOVERY_QUEUE finishes, retaining the same setup, disjoint warmup and3 measured repeats. No automatic retries or outlier deletion. The earlier preflight-only failures in other systems are not completed measurement repetitions.
