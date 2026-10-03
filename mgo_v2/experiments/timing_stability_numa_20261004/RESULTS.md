# HARNESS_STABLE

Policy timing remains paused. No automatic resume.
Guest-visible topology does not identify physical host NUMA.

## S1 monitor comparison

| Mode | n | Decode median [min, max], s | Spread |
|---|---:|---:|---:|
| HEAVY | 3 | 138.454 [93.028, 141.793] | 35.22% |
| BOUNDARY | 3 | 72.405 [71.516, 72.422] | 1.25% |

MONITOR_CONFOUND: **True**.

## S2A residency

NORMAL already reads the full CPU expert pool during load and performs a
complete warmup. PRETOUCH adds CPU reads immediately before measurement.
Two repeats per mode are descriptive; fault counts cover whole rank processes.

| Mode | Decode median [min, max], s | Spread | Minor faults, totals per run | Major faults, totals per run |
|---|---:|---:|---|---|
| NORMAL | 88.736 [72.574, 104.899] | 36.43% | [23923, 23926] | [0, 0] |
| PRETOUCH | 81.555 [71.264, 91.845] | 25.24% | [23923, 23923] | [0, 0] |

PAGE_RESIDENCY_CONFOUND: **False**; S3 uses **NORMAL**.
No disk-I/O attribution is made solely from file backing or minor faults.

## S2B H2D scaling

See `S2B_H2D.json` and the corresponding CSV for full per-device/pair distributions.

## S2C empirical SHM pairs

See `S2C_SHM.json` and the corresponding CSV for full per-device/pair distributions.

## Fixed-affinity stability

| Horizon | Env | n | Decode median [min, max], s | Spread | Gate |
|---|---|---:|---:|---:|---|
| decode64 | env1 | 3 | 70.365 [70.066, 70.686] | 0.88% | PASS |
| decode64 | env2 | 3 | 69.440 [68.940, 69.713] | 1.11% | PASS |
| decode256 | env1 | 3 | 300.293 [299.814, 302.901] | 1.03% | PASS |
| decode256 | env2 | 3 | 301.515 [299.804, 302.341] | 0.84% | PASS |

All accepted model samples pass per-rank PLAN-prefix token/cache hashes,
frozen route checks and no-compilation checks. Fault/resource reads occur
outside globally synchronized timing boundaries. Full receipts retain hashes,
affinity and safety records. H6 remains unverified; H7/H8 are screened only to
the limits of their existing logs. No policy ranking follows from this packet.
