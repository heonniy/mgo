# R4 OldCA fan-out screen

Exploratory, selected-workload single-shot screen. Positive timing gains mean faster than BR.
Confirmation observations are separate; no stability or dataset-average claim.

| Policy | Observation | H2D delta | Peer reduction | Fan-out reduction | Critical reduction | TPOT gain | Decode-wall gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| CA | 1 | 0.0816% | 17.63% | 21.57% | 9.67% | 1.48% | 1.48% |
| CA | 2 | 0.0816% | 17.63% | 21.57% | 9.67% | -0.47% | -0.47% |
| OldCA | 1 | 0.0992% | 18.00% | 28.86% | 4.09% | 1.62% | 1.62% |
| OldCA | 2 | 0.0992% | 18.00% | 28.86% | 4.09% | -1.66% | -1.66% |

See CSV files for raw seconds, E2E and all counters. Physical communication and H2D match audited CPU schedules.
No same/path affinity, A2, Env1, replication or decode256 extension.
