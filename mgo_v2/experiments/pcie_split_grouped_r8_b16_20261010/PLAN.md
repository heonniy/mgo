# R8 grouped decode: NEAR vs FAST vs NEAR_SPLIT (unbalanced PCIe split quota)

Workload: the same Qwen3-30B ShareGPT R8/C30/local-B16/input512/output64 cell as
`pcie_quota_grouped_r8_b16_20261010` (`new_OURS`: native prefill and experts,
compiled dense/layout, prefetch OFF, grouped hit-then-miss decode), with the
frozen Near teacher tokens (`pcie_quota_r8_b16_20261010/FROZEN_TOKENS.json`).

Policies differ only in the per-layer miss quota; Near expert placement inside
the quota is unchanged.

- NEAR (`LA_CA_NEAR`): balanced, extras to low rank ids. Baseline for all comparisons.
- FAST (`NEAR_FAST`): balanced, extras to the fastest measured ranks (GPUs 4-7).
- SPLIT (`NEAR_SPLIT`): unbalanced. GPUs 4-7 take about 1.25x the copies of 0-3,
  from `../pcie_haq_replay_20261010/SPLIT_LOOKUP.json`. That table is built from the
  measured group-uniform H2D grid. The microbenchmark validation
  (`validate_bench.json`) gave -10.6% H2D vs NEAR (FAST -5.0%), averaged over 48-96 misses.

Protocol: one smoke for SPLIT, then three rounds with rotated policy order
(N,F,S / F,S,N / S,N,F). Each job runs two unfiltered target repeats, giving six
samples per policy. No sample is dropped. Report TPOT/E2E median and full range
versus NEAR, plus per-rank copy totals and token differences.
