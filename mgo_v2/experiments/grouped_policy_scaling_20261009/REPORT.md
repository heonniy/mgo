# R4 Qwen main_OURS: expert scheduling, overhead and placement scaling

Strict hit-then-miss grouped execution lowered clean Near TPOT versus the
current C++ Ready-First path by 34.6% at B8, 30.8% at B16 and 33.7% at
B64. B8/B16 Near output and cache state matched between paths; B64 output
diverged under changed BF16 execution order. The grouped-path BR, CA_NATIVE
and Near policy ranges overlap at B16/B64, so there is no reliable policy
winner there. The detailed partition shows the per-expert execution span
collapsing, while exposed H2D completion waiting grows markedly with batch.

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

B8 uses the first eight rank-local requests from the B16 manifests. B16 and
B64 use existing frozen main-table manifests that contain a nested global
request set but remap some requests to different origin ranks. Therefore
B8→B16 is a cleaner local-batch scaling comparison than B16→B64; the latter
also includes the changed rank assignment. Each policy within a given batch
uses exactly the same input manifest.

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
three-kernel grouped math **launch scope** is about 14 ms/token while the
surrounding expert-execution/preparation span is about 43 ms/token. The
launch scope is a CUDA-event current-stream interval around the Triton
calls; it is not isolated kernel service. This optimization is
inferred from our profile, not implemented by the paper. The paper's
replication/admission threshold must be recalibrated with this server's
roughly 0.19–0.20 ms/expert fetch and actual grouped token-row compute;
its performance numbers cannot be transferred directly to this model/GPU.

## Batch and policy scaling

The guarded B8/B16/B64 × BR/CA_NATIVE/Near primary and diagnostic sweep
passed all planned cells. CA_NATIVE is the optimized implementation of CA
in this comparison. Every cell below is reported from its own measured
receipts.

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

### B64: completed grouped-policy primary timing

| Grouped N policy | TTFT samples (s) | TPOT samples (s/token) | Mean TPOT | E2E samples (s) | TPOT repeat difference |
|---|---:|---:|---:|---:|---:|
| BR | 5.531 / 4.095 | 0.3942 / 0.3935 | 0.3938 | 30.364 / 28.885 | 0.2% |
| CA_NATIVE | 5.754 / 4.210 | 0.3889 / 0.4052 | 0.3970 | 30.255 / 29.737 | 4.1% |
| Near | 5.733 / 4.104 | 0.3999 / 0.3843 | 0.3921 | 30.926 / 28.316 | 4.0% |

These three ranges overlap. Near has the lowest observed two-sample mean,
but the mean gap to BR is only 0.4%, far below the Near/CA repeat spread;
there is no reliable B64 policy winner from these measurements. BR has a
particularly stable pair. The separate B64 N Near run **without** compiled
dense weights above averaged 0.4087 s/token; this opt-in compiled-dense N
Near run averaged 0.3921 s/token, 4.1% lower, with matching output/cache
hashes between the two N configurations. Because those runs were at
different times, this is a measured candidate effect, not an interleaved
factorial estimate. Both grouped N configurations diverge from A-current's
generated tokens (first difference at output step 2 across the 64 local
requests), so the A→N B64 primary TPOT difference is an end-to-end
live-inference comparison rather than an exact-route scheduling contrast.

The final Near full-decode diagnostic passed. At B64, CA_NATIVE transfers
7.4% fewer off-rank bytes than BR but raises the maximum rank's expert-row
load 4.9% and adds 7.9 ms/token to the average controller CPU span. The
demand-miss count changes by only 0.4%. All three policies have distinct
output/cache hashes, so these are live-system associations.

## What scales with batch

The following records are full-decode diagnostic counts over all 63 decode
intervals, four ranks and 48 MoE layers. `MAIN hit %` counts distinct
expert uses once per layer/step; it is **not** token-row weighted. H2D bytes
are exact tagged demand-copy bytes. Off-rank bytes sum forward dispatch and
return/combine payloads. Controller, copy service and exposed wait use
different timing domains: copy service is on the overlapping copy stream,
whereas controller and wait appear in the instrumented current-stream
partition. Do not add copy service to TPOT.

| B | Grouped policy | MAIN hit % | Demand copies | Four-rank H2D GiB | Four-rank off-rank GiB | Maximum rank token-expert rows / rank mean | Controller CPU ms/token | GPU0 copy service ms/token | GPU0 exposed H2D wait ms/token | GPU0 grouped math scope ms/token |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | BR | 45.4 | 127,876 | 1,123.9 | 2.01 | 199,546 / 1.03× | 7.3 | 115.4 | 55.5 | 12.0 |
| 8 | CA_NATIVE | 45.5 | 127,648 | 1,121.9 | 1.67 | 228,581 / 1.18× | 10.1 | 110.6 | 22.9 | 13.1 |
| 8 | Near | 45.4 | 127,977 | 1,124.8 | 1.92 | 197,407 / 1.02× | 8.0 | 106.9 | 17.3 | 12.9 |
| 16 | BR | 38.6 | 172,689 | 1,517.8 | 4.02 | 402,408 / 1.04× | 8.9 | 142.0 | 47.5 | 14.7 |
| 16 | CA_NATIVE | 38.6 | 172,439 | 1,515.6 | 3.49 | 439,318 / 1.13× | 13.3 | 134.9 | 43.3 | 14.9 |
| 16 | Near | 38.5 | 173,489 | 1,524.8 | 3.88 | 393,688 / 1.02× | 9.7 | 142.6 | 52.9 | 14.1 |
| 64 | BR | 31.4 | 236,268 | 2,076.6 | 16.01 | 1,582,148 / 1.02× | 12.8 | 188.5 | 90.6 | 21.3 |
| 64 | CA_NATIVE | 31.4 | 235,389 | 2,068.8 | 14.82 | 1,659,622 / 1.07× | 20.7 | 197.8 | 137.7 | 13.5 |
| 64 | Near | 31.4 | 235,734 | 2,071.9 | 15.89 | 1,571,090 / 1.01× | 13.8 | 199.5 | 138.8 | 13.5 |

In Near, distinct MAIN hit rate falls 45.4% → 38.5% → 31.4% as batch
grows, while the **token-row-weighted** hit share remains 67.1% → 67.0%
→ 67.2%. The number of distinct hits remains near 107–109 thousand,
while misses rise 128 → 173 → 236 thousand. Hot resident experts still
handle roughly two-thirds of the token rows, making a hit-only first wave
substantial even at B64. H2D demand grows only 1.35× from B8 to B16 and
1.36× from B16 to B64 despite token rows growing 2× and 4×, respectively:
each layer/step's distinct expert set saturates. Off-rank bytes grow about
2.0× then 4.1×. The B16→B64 workload also remaps origin ranks, as noted
above, so those factors are observed workload scaling, not an isolated
batch-size experiment.

### Rank-local load under the three policies

These are exact token-expert rows over the diagnostic decode. `Max/mean`
captures load concentration; it does not include different H2D and
collective waiting, so it is not itself a latency prediction.

| B | Policy | GPU0 rows | GPU1 rows | GPU4 rows | GPU5 rows | Max/mean |
|---:|---|---:|---:|---:|---:|---:|
| 8 | BR | 191,931 | 199,546 | 191,862 | 190,805 | 1.03× |
| 8 | CA_NATIVE | 183,552 | 195,290 | 228,581 | 166,721 | 1.18× |
| 8 | Near | 197,407 | 191,567 | 194,747 | 190,423 | 1.02× |
| 16 | BR | 379,537 | 402,408 | 388,279 | 378,064 | 1.04× |
| 16 | CA_NATIVE | 340,207 | 372,850 | 439,318 | 395,913 | 1.13× |
| 16 | Near | 393,688 | 387,289 | 385,621 | 381,690 | 1.02× |
| 64 | BR | 1,505,751 | 1,525,627 | 1,579,626 | 1,582,148 | 1.02× |
| 64 | CA_NATIVE | 1,392,998 | 1,659,622 | 1,510,009 | 1,630,523 | 1.07× |
| 64 | Near | 1,571,090 | 1,532,051 | 1,529,350 | 1,560,661 | 1.01× |

CA_NATIVE's communication reduction is 17%, 13% and 7% versus BR at
B8/B16/B64, respectively. It repeatedly moves more token rows to its
hottest rank and costs about 2.8, 4.4 and 7.9 ms/token more controller CPU
time than BR. At B8, both CA primary TPOT samples are slower than both BR
samples. At B16 and B64, policy sample ranges overlap and do not establish
a reliable winner. Near keeps the row distribution at roughly 1.01–1.02×
max/mean, but is likewise not a statistically secure winner over BR at
B16/B64. Minimizing communication bytes alone is not sufficient for this
measured workload.

### Full TPOT partition: same physical GPU, separate diagnostic runs

For a stable rank reference, the table below uses **physical GPU 0** in
every cell, even when another rank has the longest expert-execution span.
Each column partitions that cell's separately instrumented TPOT into
mutually exclusive current-stream intervals. A is Ready-First individual
GEMM; N is strict hit-then-miss grouped GEMM with compiled dense routing.
The A B64 column is the earlier current-main_OURS diagnostic on the same
frozen workload. Differences between columns are descriptive, not an
additive reconstruction of uninstrumented primary TPOT.

| GPU0 component (ms/token) | A B8 | N B8 | A B16 | N B16 | A B64 | N B64 |
|---|---:|---:|---:|---:|---:|---:|
| Attention/dense/output | 52.1 | 54.7 | 50.8 | 33.2 | 51.1 | 53.9 |
| Router gate | 10.0 | 10.1 | 9.8 | 9.2 | 9.8 | 9.8 |
| Routing metadata | 38.8 | 35.9 | 43.1 | 64.6 | 48.7 | 40.7 |
| Placement/index | 25.9 | 26.0 | 28.7 | 28.7 | 35.7 | 36.1 |
| Routing-weight tensor | 0.0 | 3.8 | 0.0 | 3.8 | 0.0 | 3.9 |
| H2D submission | 6.9 | 6.1 | 13.1 | 7.5 | 14.4 | 7.2 |
| Exposed H2D completion wait | 0.0 | 17.3 | 0.0 | 52.9 | 0.0 | 138.8 |
| Dispatch/forward | 76.0 | 66.3 | 71.4 | 72.0 | 78.8 | 63.5 |
| Expert execution and preparation | 202.4 | 51.7 | 243.8 | 51.4 | 308.0 | 50.4 |
| Return/combine | 44.7 | 50.2 | 54.7 | 72.6 | 54.9 | 67.1 |
| Other MoE runtime | 53.8 | 28.2 | 50.2 | 33.7 | 55.3 | 34.0 |
| **Instrumented TPOT** | **510.7** | **350.5** | **565.6** | **429.6** | **656.8** | **505.4** |

The same N Near GPU0 partition as percentages of its instrumented TPOT:

| Component share | B8 | B16 | B64 |
|---|---:|---:|---:|
| Attention/dense/output | 15.6% | 7.7% | 10.7% |
| Router gate | 2.9% | 2.2% | 1.9% |
| Routing metadata | 10.2% | 15.0% | 8.1% |
| Placement/index | 7.4% | 6.7% | 7.1% |
| Routing-weight tensor | 1.1% | 0.9% | 0.8% |
| H2D submission | 1.7% | 1.7% | 1.4% |
| Exposed H2D completion wait | 4.9% | 12.3% | 27.5% |
| Dispatch/forward | 18.9% | 16.8% | 12.6% |
| Expert execution and preparation | 14.8% | 12.0% | 10.0% |
| Return/combine | 14.3% | 16.9% | 13.3% |
| Other MoE runtime | 8.1% | 7.8% | 6.7% |

The older Ready-First path spends progressively more of its instrumented
TPOT in the expert section as batch grows: 202 → 244 → 308 ms/token.
With grouping, that section stays near 50–52 ms/token on GPU 0. Its
grouped math launch scope is about 12.9, 14.1 and 13.5 ms/token under
Near; the remainder is wave-local metadata, input gathering, weighting,
scatter and related preparation. The new explicit H2D completion wait grows
17 → 53 → 139 ms/token. This is the clearest measured shift in the
critical path as batch grows. It is **not** equal to copy-stream service,
which is 107 → 143 → 200 ms/token on GPU 0 and may overlap the current
stream. The instrumented metadata and collective spans can move up or down
with peer arrival and diagnostic overhead; their nonmonotonic values must
not be read as a change in raw NCCL bandwidth or CPU parse cost.

At B64, the N Near GPU0 partition can be opened one level further:

| Parent segment | Exclusive subsegment | ms/token |
|---|---|---:|
| Routing metadata | MoE metadata wrapper | 5.5 |
| Routing metadata | GPU packet packing | 7.1 |
| Routing metadata | Rank all-gather, including peer wait | 12.0 |
| Routing metadata | Device-to-host copy | 1.6 |
| Routing metadata | CPU parse | 3.9 |
| Routing metadata | Gate-history update | 9.0 |
| Routing metadata | Demand histogram | 1.6 |
| Placement/index | Current-controller wrapper | 4.3 |
| Placement/index | Placement controller CPU | 14.2 |
| Placement/index | Layout CPU | 7.5 |
| Placement/index | Layout GPU materialization | 10.1 |
| Dispatch/forward | Forward all-to-all span | 36.1 |
| Dispatch/forward | Token all-to-all submission | 11.9 |
| Dispatch/forward | Forward completion/wait | 15.5 |
| Expert execution | Wave preparation / gather / scatter | 36.9 |
| Expert execution | Grouped math launch scope (three Triton kernels) | 13.5 |
| Return/combine | Return all-to-all span | 37.5 |
| Return/combine | Returned-token processing | 29.6 |

These fine subsegments partition their named parent segment, subject to
rounding. The all-to-all spans include rank rendezvous and host-side launch
time. The return-processing span includes unpack/combination work and
waiting; neither row is pure network wire time. `Other MoE runtime` is an
unclassified 34.0 ms/token in the same B64 GPU0 diagnostic. It cannot be
assigned wholly to Python, CPU communication or GPU compute without more
instrumentation. Routing metadata + placement/index + other MoE runtime
sum to about 111 ms/token (22% of this diagnostic TPOT), so there is
potential headroom outside GEMM and H2D; no additional optimization is
claimed in this study.

The raw GPU0 B64 Near diagnostic also records wall and thread-CPU clocks.
The placement-controller scope is 14.2 ms/token by both current-stream
events and thread-CPU time, supporting that controller overhead as real
host work. By contrast, metadata rank all-gather is 12.0 ms/token by
current-stream events but 5.4 ms/token of thread CPU, and returned-token
processing is 29.6 versus 7.1 ms/token. Those differences show why their
larger phase spans cannot be assigned entirely to CPU parsing or useful
communication. The explicit H2D-completion wait spans 138.8 ms/token on
the current stream and about 139.1 ms/token of wall time, including
119.0 ms/token of thread CPU; the exact driver-level cause of that CPU
activity is not isolated here.

### Near: per-rank fetch and communication load

The table makes the actual work behind the Near rows explicit. Off-rank MiB
is the rank's forward plus return transport over the full diagnostic decode;
copy-stream service is independent and overlapping. The BR/CA rank records
are retained in the eleven machine-readable reports named in the receipts
below.

| B | GPU | MAIN hits / distinct uses | Demand misses | Token-expert rows | H2D GiB | Copy service ms/token | Off-rank MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 26,715 / 59,805 (44.7%) | 33,090 | 197,407 | 290.8 | 106.9 | 484.1 |
| 8 | 1 | 26,596 / 58,972 (45.1%) | 32,376 | 191,567 | 284.6 | 100.8 | 486.9 |
| 8 | 4 | 26,592 / 58,243 (45.7%) | 31,651 | 194,747 | 278.2 | 94.5 | 500.1 |
| 8 | 5 | 26,648 / 57,508 (46.3%) | 30,860 | 190,423 | 271.2 | 99.2 | 493.2 |
| 16 | 0 | 27,115 / 71,601 (37.9%) | 44,486 | 393,688 | 391.0 | 142.6 | 978.8 |
| 16 | 1 | 27,172 / 70,937 (38.3%) | 43,765 | 387,289 | 384.7 | 148.5 | 991.5 |
| 16 | 4 | 27,182 / 70,172 (38.7%) | 42,990 | 385,621 | 377.8 | 133.9 | 1,008.1 |
| 16 | 5 | 27,092 / 69,340 (39.1%) | 42,248 | 381,690 | 371.3 | 139.4 | 997.3 |
| 64 | 0 | 27,096 / 87,180 (31.1%) | 60,084 | 1,571,090 | 528.1 | 199.5 | 4,081.2 |
| 64 | 1 | 27,048 / 86,388 (31.3%) | 59,340 | 1,532,051 | 521.5 | 195.3 | 4,079.3 |
| 64 | 4 | 27,027 / 85,566 (31.6%) | 58,539 | 1,529,350 | 514.5 | 191.9 | 4,057.9 |
| 64 | 5 | 26,913 / 84,684 (31.8%) | 57,771 | 1,560,661 | 507.8 | 176.5 | 4,050.2 |

GPU 0 has the most distinct demand misses and H2D bytes at all three Near
batches, but its token-row lead is small by B64, and GPU 5's local rows
are nearly as large. A rank's measured elapsed time is also affected by
peer rendezvous. For example, B64 BR's explicit H2D waits varied across
ranks while its two primary TPOT samples remained within 0.2%; the
critical rank is not necessarily the rank with the largest standalone
copy-service sum. CA's hottest row rank is GPU 4 at B8/B16 and GPU 1 at
B64, whereas Near stays close to balanced. The placement problem is thus
jointly about row load, fetch readiness, controller work and peer arrival,
not a single rank's miss count.

## Interpretation and validation boundary

The code-path change with the largest measured effect is grouped expert
execution. On B8/B16, A and N Near preserve exact tokens/cache state; the
new N path lowers mean primary TPOT 34.6% and 30.8%, respectively, although
its opt-in compiled dense kernel also contributes. At B64, A-current to
compiled-dense N Near lowers mean primary TPOT from 0.5917 to 0.3921
s/token (33.7%), but output and later routing diverge after BF16 execution
order changes. In the isolated B64 A comparison, compiled gate-history and
layout changes accounted for a 1.3% primary TPOT improvement. A separate
B64 A-dense comparison added 1.6%, and the no-dense N-host comparison still
showed a 30.9% improvement over A-current. Therefore the large A→N gain is
not plausibly explained by metadata optimization or compiled dense routing
alone, while its exact fixed-route magnitude at B64 is not established.

At B64, A-current's C++ Ready-First path records 86,990 distinct decode
expert uses on GPU 0, plus prefill work, as individual expert groups.
N Near schedules 3,024 layer/step events into 6,048 hit/miss grouped
waves on that rank. The measured GPU0 expert section falls from about
308 to 50 ms/token, with roughly 14 ms/token in the grouped math launch
scope (three Triton kernels plus possible submission gaps).
This supports the launch/preparation-amortization explanation. N explicitly
waits for all misses before its second wave, however, so its exposed H2D
wait grows to about 139 ms/token in the instrumented B64 run. The separate
GPU-stream-wait N amendment produced identical tokens/cache/H2D but was
2.6% slower in clean B64 primary timing; moving the wait to another stream
was not a free improvement.

Communication-minimizing CA_NATIVE does cut bytes, but its heavier rank
and longer controller span erase an observable TPOT advantage at B16/B64;
at B8 it is clearly slower than BR in the two unfiltered samples. Near
keeps row load balanced, yet its BR comparison is unresolved within repeat
variation. The data support grouping as a strong execution optimization,
and identify exposed miss H2D waiting and non-GEMM metadata/layout work as
the remaining measured headroom. They do not prove a universal best
placement policy or isolate NCCL wire time from rank waiting.

All eleven queued cells passed: two clean uninstrumented target repeats
per cell and one separate full-decode diagnostic each (22 guarded jobs in
total). Every receipt names physical GPUs 0/1/4/5, the frozen workload
hash, native executor, prefetch OFF and the chosen policy. All 22 guards
reported zero residual measured GPU processes before restoring the eight
owned model loads. Each of the eleven report partitions sums to 100%,
and every tagged decode-copy count equals its demand-miss count. Host
memory and GPU-memory/temperature guards passed throughout. Raw
per-rank JSON receipts are under
`/home/hwlee/mgo-results/grouped_policy_scaling_20261009/`; the older B64
factorial receipts are under
`/home/hwlee/mgo-results/ep_overhead_r4_b64_20261009/`.
