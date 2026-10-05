# Historical Env2: audited primary timing

R4 GPUs 0/1/4/5, local B128, frozen decode64, BF16 V3 P2/T2. Both cache groups validated SHM channels before the first measurement; no P2P/NET channels were observed. This is the historical same-host Env2, not physically NVLink-free hardware.

All eight conditions stopped after two clean repetitions. No exclusions or third repeats. Means and full ranges follow. Within-environment/cache BR is the gain denominator.

| Cache | Policy | TPOT s [full range] | E2E s [full range] | TPOT gain |
|---|---|---:|---:|---:|
| C30 | BR | 1.784403 [1.783281, 1.785525] | 125.069 [125.000, 125.138] | +0.000% |
| C30 | OLD_CA | 1.847691 [1.847592, 1.847790] | 129.670 [129.616, 129.723] | -3.547% |
| C30 | FCA | 1.926511 [1.926014, 1.927008] | 136.074 [135.998, 136.151] | -7.964% |
| C30 | LA_CA | 1.773027 [1.772060, 1.773995] | 125.344 [125.227, 125.460] | +0.638% |
| C60 | BR | 1.396235 [1.396123, 1.396347] | 100.240 [100.161, 100.319] | +0.000% |
| C60 | OLD_CA | 1.434747 [1.433664, 1.435831] | 103.247 [103.145, 103.349] | -2.758% |
| C60 | FCA | 1.533615 [1.531622, 1.535609] | 110.931 [110.839, 111.023] | -9.839% |
| C60 | LA_CA | 1.391008 [1.390587, 1.391430] | 100.878 [100.856, 100.901] | +0.374% |

OLD_CA and FCA are slower than BR in both cache groups. LA_CA has small positive TPOT point estimates (+0.638% C30, +0.374% C60), but both paired 95% intervals include zero. The current repetitions therefore do not establish positive LA_CA gains. Stability of repeated absolute timing is distinct from certainty in a small paired gain.

Separate Env2 mechanism captures are in progress. Do not substitute instrumented timings for the primary values above. All primary raw observations, paired intervals, and rank receipt hashes are retained in `POLICY_REGIME_TIMING_RESULTS.json`; actual copy-count reconciliation is in `POLICY_REGIME_WORKLOAD.json`.

Env1 preceded Env2 without transport-order counterbalancing. Absolute cross-environment differences may include temporal host drift and are not an isolated transport-cost estimate.
