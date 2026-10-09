# Owner priority: G-NEAR versus R-NEAR TPOT optimization

The owner confirmed that grouped placement / naive placement means G-NEAR /
R-NEAR. Pause unfinished baseline preparation and Stage-II live timing; the
interrupted Stage-II attempt completed warmups only and supplies no primaries.
The original complete-experiment goal remains active.

Run a common six-arm cohort: R/G native expert calls, R/G all-ready grouped
decode, and R/G grouped decode with native metadata processing. Prefill remains
native for all arms. Preserve the frozen workload, cold caches, C30/P2 physical
budget, exact Near controller and quotas, no prefetch, full NUMA-shared 54 GiB
pinned source per node, and globally serialized phase order. Grouped control
indices transfer during PLAN preparation; expert H2D never overlaps token A2A
or compute. No expert-weight duplication or D2D materialization.

All arms warm on the disjoint full64 workload, then receive three counterordered
full64 unprofiled primaries. Add two repeats to every arm if any TPOT spread is
above 5%, before separate diagnostics. Require deterministic per-arm tokens and
cache state, finite logits, unchanged Torch compilation counters and Triton JIT
specialization sets in every primary. CPU native metadata must match original
FP64 accumulation and FP32 Gate scores exactly. Warmup validates exact GPU wire
bytes and grouped weighted outputs against native (relative L2 <= 1%). Grouped
versus grouped+metadata must match generated tokens, full policy trace and cache
state exactly. Report any native/grouped greedy trajectory differences rather
than requiring identical tokens across different GEMM reductions.

After primaries, collect timers-only diagnostics (no Torch profiler or
record_function contexts). Partition TPOT by global rank-completion frontiers;
report metadata, PLAN total, controller, layout and index preparation as nested
CPU spans. PLAN residual includes checksum exchange, slot binding and glue;
do not call the entire residual checksum cost. Preserve communication checks.
Compare placement gain separately at each common implementation level. Report
negative or small results as measured; the placement benefit is not assumed.

Commit each validated completed arm and the final comparison separately.

The launched cohort is stored under
`/data2/esjung/mgo-results/pcie_topology_ablation_20261009/G4_tpot_optimization_attempt1`.
After the bounded launcher reports PASS, run
`scripts/pcie_tpot_optimization_report.py --root <cohort> --out <new-report-dir>`.
Inspect the generated PNG/PDF and conclusions, then run
`scripts/archive_pcie_optimization.py --root <cohort> --report <new-report-dir> --commit`.
The archiver preserves original and stored SHA256, uses lossless gzip for large
raw files, and creates one local commit per completed arm plus a report commit.
Do not invoke it for an incomplete cohort or skip diagnosis of a failed gate.
