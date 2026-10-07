# Native CA versus main_OURS Near

**Decision: keep Near in main_OURS.** CA reduced physical peer traffic but
raised TPOT on both clean repeats. The selected runtime remains native C++
expert execution, compiled prefill/decode index paths, Near placement and
prefetch OFF. No production policy change was made.

R4/C30, local B16 (global64), input256, output64, GPUs0/1/4/5. A single
unrestricted Near capture supplied the same frozen routing and teacher tokens
to both policies. Each arm started with a cold cache, 1,835 MAIN slots and
eight reserved but unused prefetch slots. The full-pinned CPU source, BF16
partial accumulation, overlap/barriers and native executor were identical.
The two uninstrumented primaries per policy ran Near/CA/CA/Near with 63
decode forwards. All samples are retained.

| Policy | TTFT samples, s | TPOT samples, s/token | TPOT mean, s | E2E samples, s | Decode peer traffic | Decode H2D | Decode MAIN hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Near | 1.403847 / 1.400531 | 0.500617 / 0.501766 | **0.501191** | 32.942708 / 33.011777 | 3.898 GiB | 1,503.501 GiB | 38.876% |
| CA | 1.874438 / 1.450840 | 0.518066 / 0.520022 | 0.519044 | 34.512624 / 34.212216 | **3.525 GiB** | 1,506.577 GiB | 38.751% |

CA is **17.853ms/token (3.56%) slower** despite **9.56% less peer traffic**.
It performs 350 more mandatory copies across the full run and 0.20% more
H2D bytes. Neither policy issued a prefetch or reported an explicit slot wait
on any rank. Within-policy final cache/role/controller and byte receipts match
across repeats; no recompilation or nonfinite logits occurred. Relative to
the Near reference, CA differs at 38/4,096 output token positions (99.072%
agreement). These BF16 policy-dependent differences are reported, not used
to discard a timing sample. CA's first TTFT is visibly high; the TPOT result
does not depend on that first-token excursion.

One separately instrumented 16-decode-prefix pass per policy passed token
prefix parity against its corresponding primary. These are diagnostic
current-stream completion spans, **not** primary TPOT or pure kernel times.

| Metric in diagnostic prefix | Near | CA | Observation |
|---|---:|---:|---|
| Placement controller, rank mean | 9.22 ms/token | 14.07 ms/token | CA costs +4.84 ms |
| Return all-to-all completion, rank mean | 32.05 ms/token | 41.82 ms/token | CA costs +9.77 ms |
| Forward completion, rank mean | 16.86 ms/token | 16.75 ms/token | Essentially unchanged |
| Expert-compute span, slowest rank after 16-step summation | 223.68 ms/token | 222.36 ms/token | Hides changes of critical rank by layer |
| Mean per-step maximum expert token rows | 6,247 | 7,166 | CA concentrates +14.7% more rows |
| Mean per-step row range across ranks | 210 | 1,679 | CA has about 8× more row skew |
| Distinct expert groups, all ranks | 65,067 | 65,067 | Same total group work |
| Mandatory copies, all ranks | 38,128 | 38,263 | CA +135 in prefix |

CA's Hungarian assignment maximizes local effective expert-route demand
under the same balanced miss-count quota, while Near restricts choices to
within 2% of the best projected critical-rank row load. In this prefix CA
assigns 113,803 expert token rows to GPU4 versus 98,631 under Near, while
GPU0 drops from 99,545 to 89,936. This explains why lower total communication
does not imply balanced rank work. The aggregate slowest-rank row above is
**not** the layer-by-layer critical path: the slowest rank changes between
layers, so summing by rank first concealed a real critical-expert increase.

## Matched layer-event attribution from the saved diagnostic

The [event summary](EVENT_ATTRIBUTION.json) reuses the existing eight raw
diagnostic files, with no new GPU run. It aligns all four ranks at each of
16 × 48 decode layer events. Each event has one recorded expert-compute and
return-exchange span per rank; the routed expert-row total is exactly 512
per event in both policies.

| Mean per layer event | Near | CA | CA − Near |
|---|---:|---:|---:|
| Slowest rank's expert-compute completion | 4.981 ms | 5.135 ms | +0.154 ms |
| Expert-compute rank spread | 0.919 ms | 1.231 ms | +0.312 ms |
| Largest rank expert groups | 23.32 | 24.28 | +0.96 |
| Expert-group rank spread | 4.23 | 6.04 | +1.81 |
| Largest rank token rows | 133.85 | 172.40 | +38.56 |
| Return-exchange rank spread | 1.060 ms | 1.421 ms | +0.361 ms |
| Shortest return-exchange span | 0.154 ms | 0.154 ms | approximately zero |
| Return-exchange host CPU time, rank mean | 0.152 ms | 0.151 ms | approximately zero |

The +0.154ms/layer critical expert difference corresponds to 7.39ms across
48 layers per decode token, even though the slowest *rank-total* expert time
appears unchanged. Across matched events, the CA−Near increase in critical
expert time correlates with the increase in mean return span (Pearson
**0.824**, descriptive). Within each event, the cross-rank expert-versus-return
correlation has median **−0.992** in CA (−0.984 in Near): ranks that finish
expert work earlier spend longer in return. In CA, the slowest expert rank is
also the fastest return rank in **91.5%** of events. These are strong signs
that CA's concentrated work makes other ranks wait for its arrival at the
return collective. Larger local-demand assignment also concentrates native
expert *groups*: their within-layer cross-rank correlation with expert span
has median **0.985** in CA, versus **0.785** for token rows. Group launch
count is a better proximate explanation of the observed executor span than
row volume alone.

The minimum return span across ranks stays about 0.154ms/event, while the
rank spread grows. That pattern and unchanged return-call host CPU time argue
against a larger basic transfer cost as the main reason for CA's longer return
span; the physical peer bytes actually fall. The minimum is only a service
floor proxy: these CUDA-event spans cannot precisely separate NCCL transfer
from rank-arrival waiting. The 768 layer events are one trace, not 768
independent timing repetitions.

The strongest observed costs are CA's controller work, increased per-layer
critical expert service, and longer return collective completion on every
rank. Return completion includes network transfer, host submission and
waiting for peer arrivals; the diagnostic cannot apportion those
subcomponents exactly. Metadata completion also rises
51.61→58.02ms/token even though raw routing metadata is fixed, consistent
with altered rank synchronization rather than more metadata bytes. These
overlapping diagnostic spans cannot be summed into the clean 17.853ms TPOT
gap. The physical conclusion is narrower and solid: **CA lowers bytes but
loses TPOT in this main_OURS B16 cell**, so it is not promoted.

The [rank summary](RANK_DIAGNOSTIC.json) contains every rank/step count,
phase span, primary sample and raw diagnostic hash. Raw receipts remain at
`/home/hwlee/mgo-results/headline_r4_20261007/native_policy_ca_near_off_b16_20261008`.
The supervisor's minimum sampled available host memory was 1,387.2GiB and
maximum owned-GPU use 14,645MiB. It restored model loads on GPUs0/1/4/5;
GPUs2/3/6/7 were untouched. Results for other batches and cache budgets
remain unmeasured under this native/off CA comparison.
