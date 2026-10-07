# Native CA controller result

`CA_NATIVE` is an **opt-in** replacement for CA's current-layer admission
solver. It changes only the assignment algorithm: a small C++ min-cost-flow
graph directly enforces rank quotas while maximizing the same integer
local-demand objective as expanded-slot Numba Hungarian. The rest of the
selected native expert/full-pinned/compiled-layout/prefetch-OFF runtime is
unchanged. Near remains selected for `main_OURS`.

CPU validation passed 640 bounded matrices over R1/R2/R4/R8 and 0–128
misses: every native solution had the same objective and quota counts as
legacy CA, and repeated calls were deterministic. On a synthetic 60-miss
R4 case, the assignment alone took about 0.070 ms/event versus 0.116 ms
for legacy Hungarian on this host. The objective can have ties, so expert
placements are not guaranteed identical. One full policy event confirmed
equal objective and quota counts but 36 of 127 tie choices differed.

Physical R4/C30/B16/input256/output64 used GPUs 0/1/4/5. One unrestricted
Near route and teacher stream was captured, then cold-cache CA/CA_NATIVE/
CA_NATIVE/CA primaries ran on the same frozen 63-decode trace. The route
hashes also match the preceding BR/CA packet on all four ranks. Two clean
unfiltered repeats passed per arm:

| Policy | TPOT runs, s/token | Mean TPOT | Decode peer GiB | Decode H2D GiB |
|---|---:|---:|---:|---:|
| CA | 0.525355, 0.523153 | **0.524254** | 3.5253 | 1506.577 |
| CA_NATIVE | 0.518335, 0.520511 | **0.519423** | 3.5291 | 1506.639 |

`CA_NATIVE` is **4.831 ms/token (0.92%) faster** in the paired means and
faster in both repetitions. Peer bytes rose 0.109% and H2D bytes 0.0041%;
the decode MAIN hit rate changed from 38.7513% to 38.7488%. This is a small
local improvement, not enough to make CA the best policy relative to the
previous BR/Near results.

In separate instrumented 16-decode diagnostics, placement-controller CPU
span averaged **14.251 ms/token for CA** and **12.179 ms/token for
CA_NATIVE** across the four ranks, a 2.071 ms/token reduction. This is
consistent with the cheaper assignment kernel. The diagnostic is a different
run and must not be substituted for primary TPOT; the rest of the 4.831 ms
primary difference cannot be attributed solely to controller work because
tie placements, rank readiness and timing noise also change.

Within each policy, both primary repeats matched exactly on output tokens,
cache roles/state, controller counters, H2D bytes and peer bytes. Both used
the C++ native expert executor; there were no measured recompilations and
logits were finite. Across CA and CA_NATIVE, predicted tokens agreed at
**99.023%**, which reflects the different equal-objective placement path
and BF16 execution order; no bitwise cross-policy claim is made. The
instrumented prefix matched the corresponding primary tokens for each arm.

The comparison and raw receipts are in
`/home/hwlee/mgo-results/headline_r4_20261007/native_ca_controller_b16_20261008/`.
`RANK_DIAGNOSTIC.json` preserves primary rows, per-rank diagnostic spans,
cache counts and source hashes. The supervisor finished PASS and restored
owned inference loads on GPUs 0/1/4/5. `CA_NATIVE` remains opt-in because
the measured TPOT gain is modest and the tie placements change tokens.
