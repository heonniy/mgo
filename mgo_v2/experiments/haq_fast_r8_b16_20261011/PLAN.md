# R8 grouped decode: NEAR (PCIe-worst) vs FAST vs HAQ vs HAQ_FAST

Same cell and protocol as `../pcie_split_grouped_r8_b16_20261010`: grouped
hit-then-miss, prefetch OFF, frozen Near teacher tokens. NEAR is the baseline.

- NEAR (`LA_CA_NEAR`): PCIe-topology-unaware. Its balanced quota gives the
  remainder to ranks 0,1,2,..., which on this server is exactly "extras to the
  slow GPUs 0-3". It doubles as the explicit PCIe-worst balanced baseline;
  identical rows are verified by `test_original_near_matches_identical_balanced_lookup`.
- FAST (`NEAR_FAST`): balanced, with the remainder going to the fastest ranks (GPUs 4-7).
- HAQ: hit-aware water-fill quota (cap ceil(m/R)), uniform copy cost, Near inside.
- HAQ_FAST: HAQ with per-rank copy cost from in-run DMA (GPUs 0-3 250 us,
  4-7 230 us), same cap, Near inside.

Two rounds with rotated order, two target repeats per job (four samples per arm).
The `--inline-demand-h2d` setting is the same for all arms. It is recorded in
STATUS.json and chosen from the inline_h2d_r8_b16_20261011 result.
