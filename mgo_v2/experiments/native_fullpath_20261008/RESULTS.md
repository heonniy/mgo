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
