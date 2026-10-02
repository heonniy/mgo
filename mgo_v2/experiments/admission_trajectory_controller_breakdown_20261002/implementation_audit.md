# Controller implementation audit

This packet measures existing work; it does not optimize or redesign the controller. Instrumented copies preserve every original policy statement, verified by an AST identity check and stateful CPU/GPU parity gates. Normal controller/admission/eviction/runtime files and the native extension are unchanged.

Every physical rank gathers the same global metadata and independently computes the entire global plan. All 74,880 rank-event plan/cache hashes match across ranks. Four times the logical work is executed across CPU processes, but summing rank times is not an E2E critical-path estimate.

Coverage calls `keys_on_rank()` for every victim, materializing the rank cache list before removing pinned keys. Candidate gate scores and damage ranks are rebuilt in Python for every chosen victim. `_sync_coverage()` constructs global resident sets and updates affected layers. Admission uses Python token-source/base-destination loops and allocates an incoming×world rank-cost matrix, then an incoming×incoming quota-expanded matrix for SciPy. Substitution repeatedly constructs expert/source groups and searches legal anchors. The unchanged cache consistency check scans all residents each event.

Allocation and loop work is included in the surrounding spans; there is no allocator profiler. The table measures list materializations/candidate visits, not Python memory allocation counts. Diagnostic set-difference counter work is separately timed and must not be mistaken for original controller cost. No GPU synchronization occurs inside these timers.

| B | Raw source | Policy | Victims | Candidate visits | Candidates/victim | Cache keys materialized | Admission share | Cost+solver share | Coverage ranking share | Diagnostic accounting share |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | random | random | 45,632 | 20,627,789 | 452.0 | 21,026,118 | 0.20% | 0.00% | 70.25% | 10.06% |
| 4 | random | hungarian_current | 45,633 | 20,626,746 | 452.0 | 21,026,596 | 0.98% | 0.61% | 69.55% | 10.04% |
| 4 | hungarian_current | random | 46,565 | 21,046,559 | 452.0 | 21,455,996 | 0.20% | 0.00% | 70.36% | 9.99% |
| 4 | hungarian_current | hungarian_current | 46,527 | 21,027,595 | 451.9 | 21,438,476 | 0.95% | 0.59% | 69.72% | 9.92% |
| 8 | random | random | 68,131 | 30,646,150 | 449.8 | 31,392,509 | 0.16% | 0.00% | 70.85% | 10.06% |
| 8 | random | hungarian_current | 68,150 | 30,655,198 | 449.8 | 31,401,288 | 1.08% | 0.72% | 70.08% | 10.03% |
| 8 | hungarian_current | random | 64,115 | 28,864,937 | 450.2 | 29,542,181 | 0.17% | 0.00% | 70.70% | 10.02% |
| 8 | hungarian_current | hungarian_current | 64,117 | 28,866,280 | 450.2 | 29,543,108 | 1.12% | 0.74% | 69.87% | 9.96% |
| 16 | random | random | 92,791 | 41,596,348 | 448.3 | 42,754,637 | 0.14% | 0.00% | 71.03% | 10.02% |
| 16 | random | hungarian_current | 92,907 | 41,647,001 | 448.3 | 42,808,074 | 1.38% | 0.96% | 70.08% | 9.74% |
| 16 | hungarian_current | random | 93,998 | 42,135,008 | 448.3 | 43,310,729 | 0.14% | 0.00% | 71.00% | 10.05% |
| 16 | hungarian_current | hungarian_current | 94,024 | 42,145,596 | 448.2 | 43,322,730 | 1.29% | 0.90% | 70.17% | 9.90% |

Shares are medians of five per-repetition fractions of instrumented controller wall time. They are descriptive CPU diagnostics, not fractions of uninstrumented GPU generation time. [All operation counts](operation_counts.csv) include global-resident changes, changed coverage layers, matrix sizes, observed nanoseconds per candidate and per coverage-sync call. Similar work counts do not guarantee equal Python sorting/scoring cost; host scheduling and data-dependent costs remain in these spans. Even history updates consume identical input rows under matched demand but can have unequal wall times, so component-time differences alone are not an isolated causal estimate of policy CPU cost.

Potential later optimization targets are candidate scoring/ranking, repeated resident/list/set construction and replicated planning. They require a separate owner-reviewed stage; none is applied here.
