# BR/CA dispatch and return barrier ablation

**Result:** `ablation_OURS` passed on R4/C30/B16/input256/output64 with
GPUs 0/1/4/5. It uses the same selected native C++ expert executor, compiled
layouts, full-pinned CPU expert source, BF16 rank partials and prefetch OFF as
`main_OURS`. Only decode adds local completion plus an all-rank rendezvous
immediately before each token `all_to_all_single`. The normal `main_OURS`
transport path remains unchanged. Each of 63 decode forwards called exactly
3,024 dispatch and 3,024 return barriers per rank; prefill called none.

Both jobs captured the same unrestricted Near routing and teacher tokens:
all four route SHA256 hashes match. BR and CA were each measured twice in
BR/CA/CA/BR order after cold-cache reset. The two jobs match **exactly** for
each policy and rank on frozen output tokens, cache/role/controller receipt,
H2D bytes and peer bytes. No measured run recompiled; logits were finite.
BR versus CA predicted tokens agree at 99.194%, so the cross-policy numerical
difference is reported rather than treated as bitwise parity.

| Mode | Policy | TPOT runs, s/token | Mean TPOT | CA vs BR | Decode peer GiB |
|---|---|---:|---:|---:|---:|
| Normal main_OURS transport | BR | 0.507339, 0.508886 | **0.508113** | — | 4.022 |
| Normal main_OURS transport | CA | 0.520597, 0.519431 | **0.520014** | +11.902 ms/token (+2.34%) | 3.525 |
| `ablation_OURS` barriers | BR | 0.605189, 0.605434 | **0.605311** | — | 4.022 |
| `ablation_OURS` barriers | CA | 0.621206, 0.616436 | **0.618821** | +13.510 ms/token (+2.23%) | 3.525 |

CA sends **12.35% fewer peer bytes** than BR but does not improve TPOT.
The ablation adds about 97–99 ms/token to both policies because it forbids
the existing H2D/dispatch/expert overlap. It is a diagnostic mode, not a
candidate to replace the selected runtime.

Waiting for H2D before dispatch also changes *when weights become ready* for
the same native grouped executor. Rank 0's BR primary ran 6,224/6,156
native waves in normal mode versus 3,072/3,072 in the ablation (one wave
per decode layer). CA shows the same pattern. Thus the ablation's **whole
TPOT delta is not solely barrier overhead**; the rank-synchronized
collective-path comparison is its intended interpretation. `RESULTS.json`
retains wave counts for every rank and repeat.

The following is a **separate instrumented 16-decode frozen prefix**. Each
number is the sum over layers of the slowest rank's current-stream span,
divided by 16 decode forwards, in ms/token. A rank can be slowest in
different layers; these columns are not additive components of TPOT.

| Diagnostic span, ms/token | Normal BR | Normal CA | Ablated BR | Ablated CA |
|---|---:|---:|---:|---:|
| Forward collective submission/completion path | 32.56 | 33.34 | **10.67** | **10.61** |
| Return collective completion path | 70.81 | 75.25 | **4.68** | **4.90** |
| Dispatch barrier wait | — | — | 30.81 | 25.41 |
| Return barrier wait | — | — | 66.44 | 72.46 |
| Required H2D exposed wait | — | — | 109.65 | 101.10 |
| Expert execution span | 246.71 | 246.81 | 231.88 | 236.28 |
| Placement controller CPU span | 8.69 | 14.35 | 8.90 | 14.50 |

With a rendezvous immediately before return, the large normal return span
mostly moves into **return barrier wait**; the post-barrier return collective
path is only 4.68 ms/token for BR and 4.90 for CA. CA's smaller payload
therefore yields no visible collective-path saving at this packet size. CA
also spends about 5.6 ms/token more in controller work and shows about
6.0 ms/token more critical return-readiness wait in the ablation. Critical
expert execution is about 4.4 ms/token higher in the ablation diagnostic.
Those spans overlap and cannot be summed into a causal TPOT decomposition,
but they support rank arrival skew and controller cost as the main reasons
CA remains slower here, rather than a bandwidth bottleneck in return A2A.

"Collective path" is **not wire-only latency**: it includes NCCL launch,
completion and residual host/peer waiting after barrier release. The
diagnostic instruments the current stream, changes execution timing, and
must not replace the uninstrumented primary TPOT. All raw receipts remain in
`/home/hwlee/mgo-results/headline_r4_20261007/ablation_ours_br_ca_normal_b16_20261008/`
and `/home/hwlee/mgo-results/headline_r4_20261007/ablation_ours_br_ca_sync_b16_20261008/`.
`RESULTS.json` records trace hashes, diagnostic hashes, primary values and
per-rank spans; `scripts/report_comm_barrier_ablation.py` reproduces it.
