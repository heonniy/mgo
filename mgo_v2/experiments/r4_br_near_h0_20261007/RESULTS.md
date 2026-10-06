# R4 BR/Near on selected H0 full-pinned runtime

Completed eight requested primary measurements: four workloads x BR/LA_CA_NEAR, one sample each after per-policy correctness/warmup. No samples excluded or additional primary repeats. This is a single-shot comparison; repeat stability and statistical significance are unmeasured.

R4 GPUs0/1/4/5, C30 capacities [461,461,461,460], local B16/B64 (global64/256 requests), exactly256/512 prompt tokens, decode32 (33 output positions including prefill). Existing frozen BR-generated traces/teacher inputs and prior selected placement seeds are reused. H0/full-pinned, V3 overlap, P2/T2 prefetch, BF16, unique combine, async metadata. No H1b graphs, strict per-layer phase barriers, separate diagnostics or trace recapture. Prefill cache state continues into decode. Near keeps the existing LA-style predictor-based prefetch placement.

Model loading and 216 GiB rank-private CPU pinned-store construction are outside timing. Each policy starts with a reset cache; model/source pool persist across cells. TTFT/TPOT/E2E use the same wall-clock generation method as the old strict decode32 measurements, independently maximized over ranks. TPOT is decode wall /32. Runtime overlap, prefetch and source path differ from the old strict experiment, so its gains are not directly attributable to source pinning alone.

| Local batch | Input | Policy | TTFT s | TPOT s/token | E2E s |
|---:|---:|---|---:|---:|---:|
| 16 | 256 | BR | 4.095168 | 0.732371 | 27.529222 |
| 16 | 256 | LA_CA_NEAR | 4.017659 | 0.718053 | 26.993663 |
| 16 | 512 | BR | 6.455960 | 0.733857 | 29.938291 |
| 16 | 512 | LA_CA_NEAR | 6.334363 | 0.703740 | 28.852681 |
| 64 | 256 | BR | 11.935749 | 0.932610 | 41.777592 |
| 64 | 256 | LA_CA_NEAR | 11.950214 | 0.913042 | 41.166506 |
| 64 | 512 | BR | 23.803116 | 0.978999 | 55.130253 |
| 64 | 512 | LA_CA_NEAR | 23.320570 | 0.936685 | 53.292786 |

| Local batch | Input | Near TTFT reduction | Near TPOT reduction | Near E2E reduction |
|---:|---:|---:|---:|---:|
| 16 | 256 | +1.893% | +1.955% | +1.945% |
| 16 | 512 | +1.883% | +4.104% | +3.626% |
| 64 | 256 | -0.121% | +2.098% | +1.463% |
| 64 | 512 | +2.027% | +4.322% | +3.333% |

All32 measured rank receipts pass independent CPU state/role/controller/copy-bound validation, finite logits, within-policy warm token equality, and no compilation during primary measurement. BR-vs-Near token differences in warmup: 180 (BF16 reduction-order comparison; details preserved). Copies/cancellations remain recorded for every rank.

An initial attempt stopped before timing because the compact metadata path rejected batches smaller than W128. The repair permits small batches only with supplied frozen gate scores; live small-batch history still raises explicitly. Distributed metadata byte/route/gate parity and buffer-reuse tests passed at B16/B64/B128/B256 (64 checks per rank). Failure logs are retained separately.

These workloads were previously selected to expose BR-adversarial prefill headroom. They are not a representative average-case policy benchmark. One sample per policy does not establish reproducible small gains. Peak GPU bytes are process-lifetime PyTorch allocated high-water marks, not per-cell NVML peaks; RSS includes shared mapped backing. CPU pinned cost remains54 GiB/rank.
