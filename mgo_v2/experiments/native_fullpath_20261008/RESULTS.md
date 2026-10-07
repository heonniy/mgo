# Native prefill and decode follow-up

The C++ expert executor now has an explicit prefill switch. It also resolves
logical MAIN experts to physical cache slots with one C++ scan per layer,
instead of one NumPy scan per active expert. The candidate uses the already
validated compiled rank-partial layout for both prefill and decode. Routing,
placement, cache policy, P2/T2 prefetch, collectives and BF16 combine remain
the same. H0 remains the default; `native` without `--native-prefill` retains
the earlier decode-only behavior.

R4/C30 on GPUs 0,1,4,5. Each cell captures H0 routing and teacher inputs,
then runs H0/native/native/H0 on the same cold cache and 16 decode steps. The
two arms use the same prefetch setting. Primary measurements have no phase
profiling and extension compilation is outside timing. All samples are kept.

| Cell / policy / prefetch | H0 mean TTFT | Native mean TTFT | H0 mean TPOT | Native mean TPOT | H0 mean E2E | Native mean E2E |
|---|---:|---:|---:|---:|---:|---:|
| B16/L256, BR, OFF | 1.5932 | 1.3855 | 0.6438 | 0.4815 | 11.8935 | 9.0895 |
| B64/L512, BR, OFF | 4.9433 | 5.0005 | 0.8024 | 0.5784 | 17.7817 | 14.2548 |
| B16/L256, Near, P2/T2 ON | 1.6116 | 1.4169 | 0.6898 | 0.5164 | 12.6476 | 9.6788 |

Times are seconds; TPOT divides the post-first-token wall span by 16. Native
TPOT is lower by 25.21%, 27.92% and 25.14% in these three comparisons.
B16 TTFT is lower by 13.04% (BR/OFF) and 12.08% (Near/ON). B64 TTFT is
1.16% higher on the mean; this does not justify a general TTFT gain or
automatic prefill promotion for that workload.

| Cell | H0 TTFT samples | Native TTFT samples | H0 TPOT samples | Native TPOT samples |
|---|---|---|---|---|
| B16 BR/OFF | 1.595499 / 1.590901 | 1.391146 / 1.379770 | 0.648123 / 0.639416 | 0.481089 / 0.481917 |
| B64 BR/OFF | 4.784673 / 5.101893 | 5.015880 / 4.985213 | 0.795207 / 0.809597 | 0.577196 / 0.579585 |
| B16 Near/ON | 1.626498 / 1.596607 | 1.432747 / 1.401046 | 0.708439 / 0.671068 | 0.515569 / 0.517164 |

All 12 primary passes finished successfully. Every predicted token agrees
with the H0 reference on the frozen routing/teacher workload. Final cache
state/role hashes, controller counters, prefill H2D bytes, decode H2D bytes
and decode peer bytes are identical between arms for each cell. The B16 Near
run exercises P2/T2 prefetch and its slot promotions. There is no evidence of
an added H2D wait or new barrier. These checks do not establish bitwise logit
identity or unrestricted greedy-token equivalence.

The compiled extension's physical-slot resolver matches dynamic role swaps.
A CPU-only 64-group/461-slot microbenchmark measured the old NumPy searches
at 0.133 ms/event versus the new C++ scan at 0.0027 ms/event (median of five
1000-call trials). This isolated figure is not a TPOT attribution: the native
executor also changes expert FFN dispatch and kernel packaging. The placement
and layout algorithms already use Numba; a language-only C++ rewrite of them
has no demonstrated benefit. Metadata collectives and H2D/communication
remain hardware work rather than removable Python overhead.

Four-GPU smoke and full comparisons stayed within the shared-host limits.
Minimum observed host available memory was 1370.9 GiB (smoke), 1403.7 GiB
(B16 BR), 1387.8 GiB (B64 BR) and 1372.9 GiB (B16 Near). Peak sampled HBM
on the owned GPUs was 14,623 MiB for B16 and 19,213 MiB for B64. The supervisor
restored inference loads only on GPUs 0,1,4,5 after each job. It did not use
GPUs 2,3,6,7.

Raw receipts: `/home/hwlee/mgo-results/headline_r4_20261007/`
`native_full_smoke_20261008`, `native_fullpath_b16_20261008`,
`native_fullpath_b64_20261008`, and
`native_fullpath_near_on_b16_20261008`. The first B16 job was launched from
the working tree immediately before its implementation was committed as
`572949b`; its code content was unchanged during measurement. B64 ran from
`572949b`, and Near/ON from `7752589`. The explicit prefill CLI was added in
`df09865` with the same native compute path.

The original main-table H0 and legacy decode-layout defaults are preserved.
This comparison has two samples per backend and 16 decode steps, so it does
not replace a full64-step main-table panel. The B64 H0 TTFT samples differ
by 6.4%; no outlier was discarded or extra repeats added.

An additional unrestricted Near/P2/T2 greedy 64-output-token run with
`--native-prefill` and compiled layouts passed all four rank state, finite
logit and no-recompilation checks. Its one clean primary measured TTFT
3.201497s, TPOT 0.541267s and E2E 37.301344s. This single run has no
same-run H0 comparator and is not used to estimate native speedup. Raw receipt:
`native_near_greedy_b16_20261008` under the same results root.

The first Near prefetch ON/OFF attempt
(`native_near_prefetch_ab_b16_20261008`) stopped after capture, before any
primary measurement: the new diagnostic tried to stack prefill routing
(4096 local tokens) with decode routing (16 local tokens). This was a
diagnostic shape error, not an HBM or model failure. The worker now gathers
flattened routes with explicit event lengths; a variable-length regression
test passes. The failed receipt is preserved and a new label is used for the
retry.

## Near prefetch ON/OFF, native full path

The repaired retry (`native_near_prefetch_ab_b16_v2_20261008`) passed. It uses
one frozen B16/L256 route and teacher stream, 64 output tokens (63 decode
forwards), native prefill/decode and the same 1,835 MAIN + 8 reserved
prefetch slots in both modes. The only mode difference is whether P2/T2
prefetch reservations are issued. ON/OFF/OFF/ON primaries all pass within-mode
cache/role/controller and byte parity. All four ranks report zero explicit
H2D waits. Times are seconds, two samples per mode; all samples retained.

| Mode | TTFT samples / mean | TPOT samples / mean | E2E mean | Decode distinct MAIN hit | Mandatory demand copies | Prefetch copies | Total H2D |
|---|---|---|---:|---:|---:|---:|---:|
| ON | 1.429304 / 1.388430; 1.408867 | 0.549124 / 0.547434; 0.548279 | 35.950448 | 46.889% | 155,564 | 23,688 | 1,575.457 GiB |
| OFF | 1.389885 / 1.392667; 1.391276 | 0.505583 / 0.504679; 0.505131 | 33.214548 | 38.622% | 178,834 | 0 | 1,571.783 GiB |

There are 287,548 global layer-event distinct expert demands, including
281,481 in decode, measured from the frozen routing stream outside primary
timing. The hit-rate denominator is these distinct demands; the numerator is
demands that do not require a mandatory MAIN fetch after any prefetch
promotion. The whole-run effective MAIN hit rates are 45.900% ON and 37.807%
OFF. ON saves 23,270 mandatory copies but adds 23,688 speculative copies, so
physical H2D rises by 418 copies (3.674 GiB) and TPOT is 7.87% higher.
23,242/23,688 prefetches are later used. High usefulness does not make them
worthwhile here because demand H2D has no exposed wait in these passes;
prefetch controller work and background copy contention remain possible
contributors. The clean timing does not separate those two costs.

ON and OFF produce identical outputs within each mode. OFF differs from ON
at 28 of 4,096 generated token positions (99.316% agreement) despite frozen
routing and teacher inputs. They use different placement/cache trajectories,
so this is a real output difference to report. BF16 accumulation order is a
possible cause, but this run does not isolate it. Finite logits and cache
consistency pass in both modes.
For this C30/B16 Near candidate, the measured setting is **prefetch OFF**.
The main-table supervisor now selects that explicit profile; generic H0
runtime selection remains available for historical comparisons.

## BR versus Near with native and prefetch OFF

`native_policy_br_near_off_b16_20261008` holds the same B16/L256 frozen
64-output-token routing and teacher inputs, full-pinned source, compiled
layouts, native expert executor and 1,835 MAIN slots. Only rank placement
changes. BR/Near/Near/BR all pass. Within each mode, tokens, cache/role
state, controller counters and bytes match exactly. The policies differ at
40/4,096 generated token positions (99.023% agreement); this deterministic
BF16 output difference is reported rather than discarded.

| Policy | TTFT samples / mean | TPOT samples / mean | E2E mean | Decode MAIN hit | Decode peer traffic | Decode H2D |
|---|---|---|---:|---:|---:|---:|
| BR | 1.375111 / 1.382239; 1.378675 | 0.505403 / 0.506954; 0.506178 | 33.267919 | 38.883% | 4.021 GiB | 1,502.517 GiB |
| Near | 1.968751 / 1.394881; 1.681816 | 0.494821 / 0.495984; 0.495402 | 32.892170 | 38.873% | 3.903 GiB | 1,502.771 GiB |

Near's observed TPOT is 2.13% lower and peer traffic 2.93% lower, while
H2D and hit rate are essentially unchanged. This supports rank placement's
communication effect on this cell. It does not isolate communication from
expert-compute balance as the unique cause of the TPOT change. Native expert
group counts per rank also vary, and are not a token-row service-time measure.
The two Near TPOT samples differ by only 0.23%, but Near TTFT includes a
large first-sample excursion. All samples are retained and no stable TTFT
advantage is claimed. The selected B16 main-table candidate is Near/OFF;
other cells still need their own main-table measurements.

## Final-profile unrestricted generation check

The selected supervisor profile (`--ours-final`) ran one cold-cache,
unrestricted greedy R4/C30/B16/L256/O64 target successfully on GPUs
0, 1, 4 and 5. It selected native prefill and decode, both compiled layouts,
Near placement and prefetch OFF on every rank. The target measured TTFT
3.103543s, TPOT 0.491700s and E2E 34.080626s; the warmup measured
1.440135s, 0.691065s and 44.977234s respectively and is not a primary
sample. All four ranks passed finite logits, cold cache, cache/role consistency
and no-recompilation checks; controller prefetch-issued count was zero.
Receipt: `ours_final_native_near_off_b16_20261008`. The single primary does
not establish stable TTFT or a greedy BR-versus-Near speedup. The frozen
paired comparison above supplies the placement comparison.

A fresh BR/Near frozen pair with two uninstrumented 63-decode primaries per
policy and a separate rank-level 16-decode diagnostic also passed. It found
BR/Near mean TPOT 0.511853/0.501073s; the return-collective completion span
falls across every rank, while critical expert execution and H2D remain
nearly unchanged. See [RANK_DIAGNOSIS.md](RANK_DIAGNOSIS.md) and the full
[rank summary](RANK_DIAGNOSTIC.json) for the measurements and limits.
