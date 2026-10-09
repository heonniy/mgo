# main_OURS expert execution A/B/C

The owner stopped the Qwen R8 four-system table during its DeepSpeed second
target. This packet compares three **decode expert execution** choices in
main_OURS on the same frozen ShareGPT Qwen3-30B R8/C30/local-B16/input512/
output64 workload. Physical GPUs are 0–7. All arms keep Near placement,
prefetch OFF, the rank-private full-pinned source, compiled prefill/decode
layouts, native C++ prefill, the same H2D scheduler, and the same dispatch and
return collectives. Only the decode expert scheduling/backend changes:

| Arm | Decode execution |
|---|---|
| A | Existing C++ Ready-First executor; individual expert GEMMs. |
| B | Wait for all demand H2D, then one dynamic grouped GEMM wave. |
| C | Snapshot readiness after dispatch; execute hits plus already-ready misses as one grouped wave; wait for remaining misses and execute them as a second grouped wave. |

The grouped backend is the existing Triton BF16 dynamic grouped executor. It
uses the same weight slots and row/column routing as A; no expert weights are
copied GPU-to-GPU. A is the selected main_OURS runtime; B and C are explicit
experimental modes and must not silently become the default.

Run functional 8-GPU smoke for B and C, then A/B/C full jobs with one disjoint
warmup and two unfiltered measured targets each. Use the existing frozen
workload SHA and guarded launcher. Compare global TTFT, TPOT, E2E, generated
token differences, cache/H2D counters, and per-rank first/second-wave expert
counts. A/B/C output may differ numerically because GEMM accumulation order
changes; report token differences rather than claiming bitwise parity. If
TPOT or E2E differs by >2% but <=5% between the two primary repeats, run
exactly one third confirmation; if either differs by >5%, mark unstable and
retain both original measurements without open-ended repeats.

Protect all eight GPUs and host memory with the existing launcher. Run one
job at a time; restore the eight owned model-inference loads after every job.
Keep raw prompts and token arrays outside Git. Commit only validation and
aggregate findings.

## Owner R4 extension and decode breakdown

Also run A/B/C on R4 physical GPUs 0/1/4/5 with the already frozen Qwen
ShareGPT C30/local-B16/input512/output64 workload from
`qwen_cache_ablation_20261009`. Keep the other four owned model loads stopped
throughout GPU measurement to avoid background interference, restoring all
eight after each job. Use the same two-repeat stability rule and preserve all
target samples.

Separately instrument one full decode64 target per arm after a disjoint warmup.
Break down the current-stream critical path into attention/dense/router,
metadata packet pack/all-gather/device-to-host/CPU parse/gate history,
placement controller, layout and GPU tensor preparation, demand H2D submit
and exposed wait, forward dispatch/finish, expert execution, return exchange
and combine, and residual. Report per-token milliseconds and percent of the
instrumented decode critical path. H2D copy-stream service and per-rank totals
can overlap other phases and must not be added to that partition. Keep
instrumented timing separate from the unprofiled TPOT table and verify token,
cache and H2D parity against its target.
