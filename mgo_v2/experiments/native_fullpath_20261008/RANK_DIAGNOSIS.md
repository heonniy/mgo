# Why Near lowers TPOT versus BR in the selected native profile

R4/C30, local B16 (global 64), input256, output64, GPUs0/1/4/5. Both arms
use the same frozen routing and teacher tokens, cold cache, 1,835 MAIN slots,
full-pinned CPU source, native prefill/decode, compiled packet layouts and
prefetch OFF. Four uninstrumented primaries ran BR/Near/Near/BR, followed by
one separately instrumented 16-decode-prefix pass per policy. All passes
completed, including token-prefix parity between each diagnostic and its
same-policy primary. Raw receipt:
`/home/hwlee/mgo-results/headline_r4_20261007/native_policy_br_near_off_rankdiag_b16_v3_20261008`.
`RANK_DIAGNOSTIC.json` records all rank and step summaries plus raw hashes.

| Policy | TPOT samples, s | Mean TPOT, s | Decode peer traffic | Decode H2D | Decode MAIN hit |
|---|---:|---:|---:|---:|---:|
| BR | 0.511617 / 0.512089 | 0.511853 | 4.021 GiB | 1,502.517 GiB | 38.883% |
| Near | 0.500950 / 0.501196 | 0.501073 | 3.903 GiB | 1,502.771 GiB | 38.873% |

Near lowers primary TPOT by 10.780ms/token (2.11%) and peer traffic by
2.93%. H2D volume changes by only +0.017%, and MAIN hit rate by −0.010
percentage points. The two TPOT samples per arm are close; none was filtered.
Near's first TTFT is 1.989482s versus 1.406527s in its second repeat, so
this comparison does not establish a TTFT gain. Different BF16 placement
paths generate different outputs at 40 of 4,096 token positions, as in the
earlier frozen pair; each policy is deterministic within itself.

| GPU | 16-step expert token rows, BR → Near | Expert completion, ms/token, BR → Near | Return collective, ms/token, BR → Near | Peer bytes over 63 decode, MiB, BR → Near |
|---:|---:|---:|---:|---:|
| 0 | 98,250 → 99,633 | 222.79 → 222.98 | 28.35 → 19.36 | 1,028.4 → 992.1 |
| 1 | 97,806 → 97,474 | 214.73 → 213.23 | 44.34 → 37.11 | 1,029.6 → 1,001.8 |
| 4 | 99,337 → 98,875 | 223.29 → 221.37 | 34.75 → 28.60 | 1,030.4 → 1,005.0 |
| 5 | 97,823 → 97,234 | 208.77 → 207.79 | 48.78 → 41.58 | 1,029.4 → 998.3 |

The maximum expert-completion span barely changes (223.29→222.98ms/token),
and the busiest rank's total token rows do not monotonically improve. The
mean per-step maximum token-row load moves only 6,268.9→6,247.9 (−0.34%)
over the diagnostic prefix. Placement's balanced fetch quota also holds:
37,636→37,647 mandatory copies in the prefix; primary H2D is essentially
unchanged. All primary ranks report zero explicit H2D slot waits. Thus this
cell does not support expert compute rebalance or fewer PCIe misses as the
main source of the TPOT change.

The return-collective completion span falls on **all four ranks**, averaging
39.06→31.67ms/token (−7.39ms) in the separate diagnostic. Forward completion
is unchanged at 16.80→16.82ms/token. The local partial accumulation/combine
span outside the exchange is also essentially unchanged at 25.25→25.27ms.
The controller itself is slightly *more* expensive under Near:
8.38→9.27ms/token in the instrumented prefix. Near spends that placement
work to choose higher local token demand within 2% of the best projected
critical-rank row load, while BR balances fetch counts without that locality
choice. Fewer remote token packets reduce both forward and return payloads;
the measured time reduction appears in return exchange completion, where
rank arrival and collective waiting are exposed.

This identifies the **return communication/wait path** as the strongest
observed TPOT component in this B16 setting. The CUDA current-stream spans
include host submission gaps and waiting for peers, so 7.39ms is not a pure
NVLink wire-time saving and is not an additive decomposition of the clean
10.78ms primary gain. One diagnostic pass per policy cannot apportion the
return change between packet transfer and peer arrival skew. Metadata spans
also differ, but routing metadata is identical across arms and those spans
are sensitive to host scheduling; they are not evidence that Near changes
metadata work. Other batch/cache cells still require their own measurements.

The diagnostic job stayed within the shared-host guards: minimum available
host memory 1,478.6GiB, highest sampled owned-GPU use 14,645MiB. The
supervisor restored the model loads on GPUs0/1/4/5 and never used2/3/6/7.
