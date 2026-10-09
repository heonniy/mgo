# R-NEAR identical-packet grouped/native compute probe

The native model retains exactly the reference tokens and final cache on all four ranks. The alternate grouped outputs are checked but never advance the model. This is a computation diagnostic, not a grouped serving TPOT result or a Ready-First experiment.

48 real decode events (steps 1, 32, 63; layers 0, 3, ..., 45) × 3 counterordered timed pairs × 4 ranks = 1152 raw local samples. Both methods receive the same input packet, expert cache weights, routing weights and physical slots, after every rank finishes mandatory H2D. GPU event windows include host enqueue gaps. Each packet warms both methods once before timing. No Torch profiler runs during these pairs.

| Backend | Median of event/repeat max-rank local completion (ms) | Mean (ms) | p10 / p90 (ms) |
|---|---:|---:|---:|
| Native expert loop | 12.141517 | 12.843816 | 6.195400 / 19.602909 |
| All-ready grouped | 0.414458 | 0.419590 | 0.293461 / 0.513232 |

Ratio of medians: 29.295×; compute-only median latency reduction: 96.586%. This ratio is descriptive of the fixed packet set, excludes rendezvous and control/index preparation, and must not be reported as overall generation acceleration.

Numerics: all 192 packet/rank checks finite; maximum weighted-output relative L2 0.4492%, maximum absolute error 0.031250. Different BF16 reduction orders remain; actual grouped greedy generation has not been validated. Additional bounded activation workspace: 8.25 MiB/rank. No expert-weight D2D copies or changes to C30/P2 residency. The source remains actual NUMA-shared 54 GiB pinned per node. The global phase-order verifier passes all 3072 events on every rank.

The result supports a large removable execution/submission cost in this native per-expert path. The C++ loop still invokes ATen/cuBLAS and gather/activation/weight operations separately for every expert. Grouped projections combine gate/up and execute many experts per launch, then grouped down and weight/scatter, reducing CPU submissions and small-operation scheduling. The separate profiler correlates each executor's launches with its actual GPU kernels; its intrusive CPU timings are not used here.

Ready-First addresses a separate exposed wait: start resident or completed experts while other expert copies remain in flight. This probe intentionally disables that opportunity by completing all H2D first. A grouped Ready-First implementation would group the currently ready experts into waves, potentially trading earlier execution for more launch batches. Forward and metadata communication must finish before H2D, and return communication must wait for all ranks' H2D and expert compute to finish.
