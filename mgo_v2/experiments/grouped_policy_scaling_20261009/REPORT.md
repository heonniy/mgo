# R4 Qwen main_OURS: expert scheduling, overhead and placement scaling

## Workload and validity

Qwen3-30B-A3B-Instruct-2507 on physical GPUs 0/1/4/5, ShareGPT, rank-local
B8/B16/B64, 512 input tokens, 64 output tokens, C30 (461/461/461/460
physical expert slots; 459/459/459/458 are MAIN after two reserved slots per
rank), rank-private pinned host expert store, Near placement unless a
policy is named, and prefetch OFF. Each primary cell has a disjoint warmup,
cache reset, and two uninstrumented target repetitions. All samples are
retained. The eight owned background model loads are stopped during timing
and restored by the guard after each job. The workers use fixed 24-core CPU
affinity per rank, `torch.set_num_threads(2)`, `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.

For each diagnostic cell, a separate full decode run records exclusive
current-stream intervals between the first and last token-ready markers.
Within one rank these intervals sum to 100% of the **instrumented** TPOT.
The diagnostic adds event markers, so its absolute TPOT is higher than the
primary uninstrumented TPOT. Collective intervals include host launch gaps
and waits for peer ranks, not only NCCL wire time. Copy-stream service may
overlap compute and communication; it is reported separately and is **not**
added to the 100% partition. The normal execution has no explicit global
barrier immediately before each dispatch or return all-to-all, but the
collectives rendezvous the ranks.

The two code-path comparisons below use the same frozen B64 manifests and
rank-local prompts. The pre-optimization and current A runs match generated
tokens, cache state and H2D bytes on all four ranks in both repetitions.
Grouped GEMM changes BF16 association: the grouped paths diverge in generated
tokens during decode, then their later live routes and fetch totals also
change. Their primary TPOT is a real end-to-end measurement on the same
requests, **not** a pure fixed-routing scheduling delta. No outlier sample
was discarded.

## B64: before optimization, current main_OURS and new_OURS

`A-original` is the isolated pre-optimization commit `adc8f44d`; `A-current`
has compiled gate-history updates and bulk layout views while retaining the
C++ Ready-First per-expert GEMM. `A-dense` adds the Qwen-only opt-in compiled
dense routing-weight kernel. `new_OURS` groups **cache hits only** in wave 1,
then waits for all demand-miss H2D transfers and groups those misses in wave 2.
Even an already-completed miss stays in wave 2. B (`serial_all`) waits for all
H2D then uses one grouped wave and is a secondary reference.

| Path | TTFT samples (s) | TPOT samples (s/token) | Mean TPOT | E2E samples (s) |
|---|---:|---:|---:|---:|
| A-original | 5.568 / 4.086 | 0.6102 / 0.5890 | 0.5996 | 44.009 / 41.193 |
| A-current | 5.612 / 4.099 | 0.6000 / 0.5834 | 0.5917 | 43.412 / 40.855 |
| A-dense | 5.582 / 4.088 | 0.5901 / 0.5750 | 0.5825 | 42.756 / 40.310 |
| B one grouped wave | 5.769 / 4.146 | 0.4056 / 0.4114 | 0.4085 | 31.320 / 30.066 |
| new_OURS hit→miss grouped | 5.729 / 4.098 | 0.4030 / 0.4144 | 0.4087 | 31.118 / 30.202 |
| new_OURS GPU-stream-wait candidate | 5.760 / 4.113 | 0.4150 / 0.4233 | 0.4192 | 31.908 / 30.782 |

Current A improves TPOT 1.31% over original A, and the compiled-dense
candidate adds 1.56% over current A. The strict hit→miss grouped path lowers
measured TPOT 30.9% against current A. A separate stream-wait amendment
preserved **exactly** the host-wait N tokens, cache state and H2D bytes in
both repeats but raised mean primary TPOT by 2.6%, so it is not selected.
The first TTFT sample is much slower in every path; the decode-only grouped
change does not establish a TTFT improvement.

### B64 full-decode partition and fetch cost

The following is the diagnostic partition on GPU 0 for a consistent
per-rank comparison. Every column sums to about 100% of its separately
instrumented TPOT; rounding accounts for small differences. `new_OURS`
exposes the host H2D wait as its own segment, while Ready-First does not
perform that host wait and can include stream waiting inside expert execution.

| Component (ms/token; share) | A-original | A-current | new_OURS |
|---|---:|---:|---:|
| Attention/dense/output | 48.9 (7.4%) | 51.1 (7.8%) | 48.2 (9.3%) |
| Router gate | 9.7 (1.5%) | 9.8 (1.5%) | 9.5 (1.8%) |
| Routing metadata | 52.1 (7.8%) | 48.7 (7.4%) | 48.6 (9.4%) |
| Placement/index | 40.6 (6.1%) | 35.7 (5.4%) | 35.5 (6.9%) |
| H2D submission | 15.8 (2.4%) | 14.4 (2.2%) | 8.5 (1.6%) |
| Exposed H2D wait | 0.0 (0%) | 0.0 (0%) | 128.2 (24.8%) |
| Dispatch/forward | 76.7 (11.5%) | 78.8 (12.0%) | 64.7 (12.5%) |
| Expert execution and preparation | 307.5 (46.3%) | 308.0 (46.9%) | 57.1 (11.0%) |
| Return/combine | 57.3 (8.6%) | 54.9 (8.4%) | 63.5 (12.3%) |
| Other MoE runtime | 55.9 (8.4%) | 55.3 (8.4%) | 53.6 (10.4%) |
| **Instrumented total** | **664.5** | **656.8** | **517.4** |

The matched A-original→A-current path removed about 3.4 ms/token from
routing metadata and 4.9 ms/token from placement/index on GPU 0, closely
matching its 7.9 ms/token primary TPOT reduction. Grouping removes about
251 ms/token from the expert-execution segment but exposes about 128 ms/token
of host H2D waiting. The diagnostic total drops about 139 ms/token; the
uninstrumented primary mean drops 183 ms/token. Those are different
measurement modes and should not be equated term by term. B and N have
similar primary TPOT: the extra hit/miss grouped launch in N approximately
offsets the H2D overlap gained by working on hot cache hits first, within
the observed repeat variation.

The A-current diagnostic counted the following 63-step decode work:

| Physical GPU | Distinct expert uses | MAIN hits | Demand misses/copies | Token-expert rows | H2D GiB | Pure DMA service (s) | Mean per copy (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 86,990 | 27,109 | 59,881 | 1,571,428 | 526.3 | 12.01 | 0.201 |
| 1 | 86,187 | 27,058 | 59,129 | 1,533,298 | 519.7 | 11.25 | 0.190 |
| 4 | 85,441 | 27,059 | 58,382 | 1,521,860 | 513.1 | 11.02 | 0.189 |
| 5 | 84,560 | 26,920 | 57,640 | 1,566,566 | 506.6 | 10.87 | 0.189 |

These are **distinct expert uses per layer/step**, not token counts. The
distinct MAIN hit rate is only 31–32%, while 66–69% of token-expert rows
land on those hits: hot experts account for most of the compute. The pure
PCIe copy service is roughly 0.19–0.20 ms for one 9 MiB expert. Its rank
sum is about 173–191 ms per decode-token interval if divided by 63, but
copy service overlaps the current-stream path and cannot be subtracted
directly from TPOT. GPU 0 has the largest distinct load and copy volume;
GPU 5 spends more time inside metadata/return spans waiting for peers.
Collective alignment makes all ranks' end-to-end interval nearly equal, so
the rank with the largest elapsed value alone does not identify the local
bottleneck.

### TTFT

The separate one-token prefill diagnostic on GPU 0 totals about 4.18 s for
A-original, 4.17 s for A-current, and 4.16 s for N. Placement/index is
roughly 2.00/2.00/1.97 s, respectively; attention/dense is about 0.48 s,
expert execution about 0.41 s, return/combine about 0.39–0.40 s and exposed
H2D wait about 0.30 s. The optimizations studied here target decode and do
not repair the prefill placement/index bottleneck.

### HarMoEny applicability

[HarMoEny](https://arxiv.org/pdf/2506.12417) exchanges small routing IDs,
builds the same token/expert/rank schedule on each GPU (§4.1), transfers hot
expert work to less loaded GPUs only beyond a compute-versus-transfer
threshold (§4.2, §4.4), and overlaps expert fetch with earlier execution
(§4.3). OURS already exchanges routing information, makes a deterministic
multi-rank placement decision and has a separate H2D stream. Its Qwen3
top-8 plus gate-history metadata is larger than the paper's top-1 example:
at B64 each rank sends 33,304 B/layer, including 64×128 FP32 probability
values. Only the last 128 of the global 256 probability rows survive in gate
history; the probability payload from logical ranks 0 and 1 is a candidate
for omission. This must preserve the exact gate-history reduction order,
owner decisions, cache state and generated tokens, and must be timed because
an additional collective may cost more than the bytes saved. The measured
routing-metadata span includes rank and host waiting; it is not equivalent to
the wire time of a 133 KiB packet.

The more direct current `new_OURS` target is wave-local row/column index
concatenation, Python metadata-list construction, GPU metadata-tensor
creation, input gather and output weighting/scatter. On GPU 0 at B64 the
grouped GEMM kernel subspan is about 14 ms/token while the surrounding
expert-execution/preparation span is about 43 ms/token. This optimization is
inferred from our profile, not implemented by the paper. The paper's
replication/admission threshold must be recalibrated with this server's
roughly 0.19–0.20 ms/expert fetch and actual grouped token-row compute;
its performance numbers cannot be transferred directly to this model/GPU.

## Batch and policy scaling

The guarded B8/B16/B64 × BR/CA_NATIVE/Near primary and diagnostic sweep is
in progress. CA_NATIVE is the optimized implementation of CA in this
comparison. Results are added by completed batch; no unfinished cell is
inferred from another batch.

### B8: completed primary and full-decode diagnosis

| Path/policy | TTFT samples (s) | TPOT samples (s/token) | Mean TPOT | E2E samples (s) | TPOT repeat difference |
|---|---:|---:|---:|---:|---:|
| Current A, Near | 2.999 / 1.390 | 0.4349 / 0.4320 | 0.4334 | 30.396 / 28.606 | 0.7% |
| Grouped N, BR | 3.041 / 1.358 | 0.2804 / 0.2655 | 0.2729 | 20.706 / 18.083 | 5.5% |
| Grouped N, CA_NATIVE | 3.157 / 1.401 | 0.2975 / 0.2889 | 0.2932 | 21.902 / 19.603 | 2.9% |
| Grouped N, Near | 3.018 / 1.383 | 0.2992 / 0.2675 | 0.2833 | 21.865 / 18.237 | 11.2% |

Within B8 the current A and grouped N Near paths generated the same tokens
and final cache state on all ranks in both target repeats; their diagnostic
rank work, misses and transport also match. N Near's mean TPOT is 34.6%
below A Near, but N additionally enables the opt-in compiled dense routing
kernel. The B64 factorial above separates that candidate from the grouped
execution effect. The two N Near samples have an 11.2% relative difference,
so its apparent 3.8% mean disadvantage against BR is **not a stable policy
ranking**. N BR's repeat difference is 5.5%, whereas CA_NATIVE's is 2.9%.
No sample was deleted or replaced by the instrumented run.

BR, CA_NATIVE and Near do **not** have identical generated-token hashes or
final cache hashes, even though their input requests and seeds are frozen.
Consequently the policy work-count comparisons include subsequent live
routing differences; they are system-level results, not a fixed-route causal
decomposition of placement alone.

The full-decode diagnosis supplies exact work counts and a separate
instrumented phase partition:

| B8 path/policy | Four-rank MAIN demand misses / H2D copies | Four-rank H2D GiB | Four-rank off-rank forward+return GiB | Maximum rank token-expert rows | Mean controller CPU ms/token across ranks | Maximum rank copy-stream service ms/token |
|---|---:|---:|---:|---:|---:|---:|
| A Near | 127,977 | 1,124.8 | 1.92 | 197,407 | 8.0 | 103.3 |
| N BR | 127,876 | 1,123.9 | 2.01 | 199,546 | 7.3 | 115.4 |
| N CA_NATIVE | 127,648 | 1,121.9 | 1.67 | 228,581 | 10.1 | 110.6 |
| N Near | 127,977 | 1,124.8 | 1.92 | 197,407 | 8.0 | 106.9 |

CA_NATIVE lowers transported bytes 17% relative to BR, but increases the
largest rank's expert token-row load 15% and controller CPU time about
2.8 ms/token. Its H2D demand falls less than 0.2%. These measurements are
consistent with communication savings being outweighed by controller and
load cost at B8; they do **not** isolate the exact contribution to the 20 ms
primary TPOT gap. BR has the lowest observed mean primary TPOT at B8, but
the Near sample ranges overlap BR's and require cautious interpretation.

The H2D copy-stream service is a separate, overlapping duration. N Near's
two clean repeats have identical tokens and cache hashes, while GPU 0's
grouped-executor serial H2D wait fell from 2.709 to 1.617 s and GPU 4's
from 2.444 to 1.193 s. This accounts for part of its TPOT spread and
identifies a real wait variation without establishing whether host-memory,
PCIe, stream scheduling or peer arrival caused it. The diagnostic phases
also contain peer waiting: for example, the N BR anchor rank reports
74.8 ms/token in routing metadata and 88.5 ms/token in return/combine,
although its primary TPOT is faster than CA_NATIVE. Do not treat those
spans as isolated CPU parse or NCCL wire time, or compare diagnostic total
TPOT directly as the policy result.

### B16: completed primary timing

| Path/policy | TTFT samples (s) | TPOT samples (s/token) | Mean TPOT | E2E samples (s) | TPOT repeat difference |
|---|---:|---:|---:|---:|---:|
| Current A, Near | 3.646 / 1.775 | 0.4905 / 0.4910 | 0.4908 | 34.548 / 32.710 | 0.1% |
| Grouped N, BR | 3.707 / 1.738 | 0.3437 / 0.3169 | 0.3303 | 25.363 / 21.703 | 8.1% |
| Grouped N, CA_NATIVE | 3.897 / 1.770 | 0.3196 / 0.3348 | 0.3272 | 24.030 / 22.860 | 4.6% |
| Grouped N, Near | 3.678 / 1.763 | 0.3476 / 0.3320 | 0.3398 | 25.577 / 22.680 | 4.6% |

The A and N Near target repeats match generated tokens and final cache
state on every rank. Their mean TPOT differs by 30.8%, with the same
compiled-dense confound noted for B8. All three grouped policy ranges
overlap; the ordering of their means is not a stable winner claim. In N BR,
GPU 0's explicit serial H2D wait fell from 4.64 to 3.01 s between the two
otherwise token-identical target repeats. The 1.63 s difference over 63
decode intervals is about 26 ms/token, close to the 27 ms/token TPOT
difference. In N CA_NATIVE the second repeat was slower and GPU 0's wait
rose from 2.93 to 4.03 s. Thus the large repeat spread is not consistently
a first-versus-second-repeat effect. These waits expose where time varies,
not whether the DMA service itself or preceding first-wave overlap changed.

| B16 path/policy | Four-rank MAIN demand misses / H2D copies | Four-rank H2D GiB | Four-rank off-rank forward+return GiB | Maximum rank token-expert rows | Mean controller CPU ms/token across ranks | Maximum rank copy-stream service ms/token |
|---|---:|---:|---:|---:|---:|---:|
| A Near | 173,489 | 1,524.8 | 3.883 | 393,688 | 9.8 | 138.4 |
| N BR | 172,689 | 1,517.8 | 4.021 | 402,408 | 8.9 | 142.8 |
| N CA_NATIVE | 172,439 | 1,515.6 | 3.493 | 439,318 | 13.3 | 141.0 |
| N Near | 173,489 | 1,524.8 | 3.883 | 393,688 | 9.7 | 148.5 |

BR and CA show a 13% lower four-rank off-rank byte count under CA, but
CA's largest-rank token-expert row load is 9% higher, controller CPU time
is roughly 4.4 ms/token higher, and demand-miss copies are nearly unchanged.
The policies have different generated-token and cache hashes, so these are
live-system associations, not fixed-routing placement deltas. A and N Near
have exactly matching work counts and token/cache hashes. On physical GPU 0,
the diagnostic expert-execution/preparation span falls from 243.8 to
51.4 ms/token under grouping, while an explicit 52.9 ms/token H2D wait
appears and the `Other MoE runtime` span falls from 50.2 to 33.7 ms/token.
Those phase spans come from different, instrumented runs; their change is
descriptive and is not substituted for the clean primary TPOT gain.
