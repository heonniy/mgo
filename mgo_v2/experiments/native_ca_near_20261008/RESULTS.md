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
| Expert-compute span, slowest rank | 223.68 ms/token | 222.36 ms/token | No measured critical-compute increase |
| Mean per-step maximum expert token rows | 6,247 | 7,166 | CA concentrates +14.7% more rows |
| Mean per-step row range across ranks | 210 | 1,679 | CA has about 8× more row skew |
| Distinct expert groups, all ranks | 65,067 | 65,067 | Same total group work |
| Mandatory copies, all ranks | 38,128 | 38,263 | CA +135 in prefix |

CA's Hungarian assignment maximizes local effective expert-route demand
under the same balanced miss-count quota, while Near restricts choices to
within 2% of the best projected critical-rank row load. In this prefix CA
assigns 113,803 expert token rows to GPU4 versus 98,631 under Near, while
GPU0 drops from 99,545 to 89,936. This explains why lower total communication
does not imply balanced rank work. Yet the measured expert-compute completion
maximum does not rise: group launches and scheduling also matter, so the row
skew alone is **not proven** to cause the entire timing loss.

The strongest observed costs are CA's controller work and longer return
collective completion on every rank. Return completion includes network
transfer, host submission and waiting for peer arrivals; the diagnostic
cannot apportion those subcomponents. Metadata completion also rises
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
