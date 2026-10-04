# AGENT TASK — phase-aware B128/B256 policy headroom

1. Read PLAN.md and PHYSICAL_EXECUTION_ORDER.md.
2. Reuse the completed 2048-request exact MATH/ShareGPT pools. No GPU capture.
3. Run the B128/B256 CPU packet with:
   `python scripts/run_r8_phase_aware_policy_cpu.py`
   or the guarded wrapper `python scripts/run_r8_phase_aware_policy_cpu.py` through
   `scripts/run_r8_phase_aware_policy_cpu.py`.
4. The implementation-validation gate must reproduce the archived B128
   MATH sample97/dp200/placement172 BR critical rows, H2D, peer bytes and hit
   rate exactly before the new search proceeds.
5. Search two separate stress families:
   - COMM-worst random BR one-copy seed/subset/rank assignment;
   - LOAD-worst random BR one-copy seed/subset/rank assignment.
6. Freeze R8, cache60, Gate W128, substitution OFF, prefill+decode256.
7. Mandatory misses must remain balanced across ranks for every policy.
8. Evaluate BR, CA, LA, BR+REP, CA+REP and LA+REP.
9. Real replicas consume actual slots, can evict unique/duplicate victims, and
   are created by D2D from either a resident source or a current miss after its
   H2D completes.
10. Model D2D overlapping remaining H2D exactly as PLAN.md specifies and also
    report the no-overlap serial counterfactual.
11. Use existing H2D/peer/GPU-profile microbenchmarks for calibrated phase
    estimates; keep D2D as a 0.02/0.05/0.10/0.20-ms sensitivity until the new
    four-GPU model-free microbenchmark is run.
12. Report prefill, decode and MoE-total separately. TPOT interpretation is
    decode-only; prefill model excludes attention/dense time.
13. Commit results and stop. Do not automatically launch full-model GPU timing.

Optional later calibration, which does not require the offloading runtime:
`torchrun --standalone --nproc_per_node=4 examples/replica_phase_microbench.py --output <dir>`.

## Required execution order

Do not run the CPU GO packet against guessed D2D costs. First run
`scripts/run_replica_phase_microbench.py` for Env1/Env2 on four free GPUs, or
use the chained `scripts/run_replica_calibrated_cpu_packet.py`. Only after
`microbench_calibration.json` is PASS may the B128/B256 CPU replay run.
