# DeepSpeed small-cell bounded confirmation

All three repeats pass frozen workload/raw-clock, cold-parameter, GPU KV, finite-logit and byte-cap checks. TTFT remains UNSTABLE; TPOT and E2E individually satisfy5% spread. Preserve every sample, including repeat1; no stable whole-row headline selection and no automatic further repeats.

- TTFT: median 5.412408645s; range [5.320577166, 6.045086160]s; spread 12.955%.
- TPOT: median 4.085472877s; range [4.065967077, 4.095595406]s; spread 0.726%.
- E2E: median 262.797199889s; range [262.201012034, 263.343087752]s; spread 0.435%.

Fixed per-rank CPU ranges match OURS. Compared with the earlier unbound launch, decode spread is smaller, but sequential uncontrolled runs do not prove CPU affinity caused the change. This is stock ZeRO3 CPU parameter offload, not paper FastGen. All-parameter peak remains3115216896bytes/rank, below4348182528byte/rank cap.
