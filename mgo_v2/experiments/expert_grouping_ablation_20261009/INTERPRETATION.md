# What changed in main_OURS decode A/B/C

All comparisons use frozen Qwen3-30B ShareGPT local-B16/input512/decode64,
C30 and Near with prefetch OFF. A is the selected C++ Ready-First executor.
B waits for all required H2D copies, then executes one Triton dynamic grouped
expert wave per layer. C executes a ready wave before waiting for a remaining
miss wave. The B/C grouped backend is experimental and is **not** the selected
main_OURS default.

The two unprofiled R8 repeats give mean TPOT A 391.22, B 308.76 and C
307.11 ms/token. The corresponding R4 values are 499.88, 344.43 and 345.61
ms/token. B improves on A by 21.08% at R8 and 31.10% at R4. B/C differ by
only +0.54% in C's favor at R8 and -0.34% at R4; these small reversals are
inside the observed repeat variation. Both C and B produced exactly the
same 8,192 R8 and 4,096 R4 generated tokens as A in each compared repeat.
Per-rank H2D copies, bytes and final cache-state hashes also matched exactly.
R8 transferred 1,856.681 GiB in 211,249 expert copies; R4 transferred
1,578.313 GiB in 179,577 copies.

The separately instrumented R4 decode critical path identifies the source
of the large A→B gain. On the slowest diagnostic rank, expert execution and
preparation fell from **244.59 ms/token (42.3%)** in A to **25.80 ms/token
(5.9%)** in B. B paid **79.79 ms/token (18.4%)** of exposed H2D waiting after
it serialized all required copies. Even with that cost, its instrumented
critical path fell from 578.33 to 434.55 ms/token. B used one grouped wave
per decode-layer event, whereas A required multiple ready waves and launched
individual expert GEMMs. A→B changes both the GEMM backend and H2D schedule,
so this experiment cannot assign an exact fraction of the speedup to either
change alone; the measured expert section dominates the net difference.

C exposed 55.17 ms/token of H2D wait, 24.62 ms below B, because its first
wave covered 66.0% of R4 expert groups already ready after dispatch. Its
expert execution/preparation rose to 54.49 ms/token, 28.70 ms above B. The
extra grouped wave nearly erased the overlap saving. C used 24,142 grouped
waves for 12,096 R4 decode-layer events, compared with B's 12,096 waves.
At R8, the first C wave covered 63.7% of expert groups. The R4 instrumented
critical path was 433.44 ms/token, only 1.11 ms below B, consistent with the
tiny and direction-changing unprofiled B/C difference.

Routing metadata remained **52.30 / 53.57 / 53.64 ms/token** for A/B/C in
the R4 diagnostic. Its share rose from 9.0% in A to 12.3–12.4% in B/C
because the total shrank. The metadata rank all-gather alone occupied
23.48–25.78 ms/token and includes waiting for peers; gate-history work was
9.42–9.97 ms/token, GPU packet preparation 6.94–7.05 ms/token, and device-to-
host copy only 1.36–1.49 ms/token. Placement/index preparation was ~32
ms/token across arms. Metadata is a remaining optimization target, but its
cost did not cause the A/B/C TPOT difference.

Forward and return collective completion spans include rank-arrival waits,
host gaps and combination work; they are not pure network transmission.
H2D copy-stream service overlaps the current-stream partition and must not
be added to its 100%. The diagnostic itself adds markers and raises absolute
TPOT; use `PRIMARY_RESULTS.md` for performance and `BREAKDOWN.md` for phase
shares. Every first target had a slower TTFT, making E2E exceed the 5%
two-repeat stability threshold. No outliers or extra repeats were added.

**Decision from this tested condition:** B is the observed R4 winner and
B/C are effectively tied at R8. Grouped expert execution is worth retaining
as an opt-in candidate; this study does not promote it to all main_OURS
workloads without broader capacity and batch validation.
