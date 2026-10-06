# Pure LA versus LA_CA_NEAR: clean R4/B64/L512 timing

Completed exactly one clean primary measurement per policy, after one full warm correctness pass each. No phase instrumentation or additional timing repeats. Results are single-shot observations, not repeat-stability or significance evidence.

R4 GPUs 0,1,4,5; C30 capacities [461,461,461,460]; local B64 (global B256); input512; decode32 plus first output from prefill. H0/full-pinned V3 P2/T2, BF16, unique combine, async metadata. Same original frozen routes, teacher tokens and placement seed28. Policies apply to both prefill and decode, with prefill cache carried into decode. Pure LA is controller policy4; LA_CA_NEAR is policy7. Existing prefetch behavior is retained in both. No new trace capture, seed search or graph cache.

Model loading and 216 GiB CPU pinned-store construction are outside timing. Each policy starts with fresh cache state, reusing the model and source pool. Measurement order is LA then LA_CA_NEAR after both warmups. TTFT, TPOT and E2E are independently maximized over the four ranks; TPOT is decode wall time divided by32.

| Policy | TTFT s | TPOT ms/token | E2E s |
|---|---:|---:|---:|
| LA | 22.470298 | 925.007 | 52.069893 |
| LA_CA_NEAR | 22.239911 | 925.142 | 51.843648 |

Near reduction relative to pure LA: TTFT +1.025%, TPOT -0.015%, E2E +0.435%. Negative means Near was slower.

All8 measured rank receipts passed CPU state/role/controller/copy-bound checks, finite logits, within-policy warm/measurement token equality and no compilation during timing. Cross-policy output differences: 77/8448; frozen teacher inputs remain identical, and cross-policy bitwise equality is not claimed. No further numerical investigation was added, consistent with the owner BF16 scope.

Input hashes and pure-LA CPU replay were validated before launch; Near reused its existing independent reference. Distributed metadata parity preflight passed. Host memory guards remained enabled; no OOM. Owned model-forward loads restored only on0,1,4,5 after completion. GPUs2,3,6,7 were untouched.

The inherited workload/seed was selected for BR-adversarial prefill headroom. This is not an average-case LA-versus-Near benchmark. Previous BR/Near primary results and diagnostic timings are separate runs and are not used as this comparison's baseline. No timing samples were excluded.

Executed source commit: `7a2e84afeb9ebe289b9060a850fe1d43937baa72`.

Artifacts: [PRIMARY_SAMPLES.csv](PRIMARY_SAMPLES.csv), [RANK_SAMPLES.csv](RANK_SAMPLES.csv), [INPUTS_AND_CPU_PROOFS.json](INPUTS_AND_CPU_PROOFS.json), [SOURCE_HASHES.json](SOURCE_HASHES.json), and all warm/primary rank receipts in [raw/](raw/).
