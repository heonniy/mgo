# AGENT TASK — physical E2E / TPOT offloading

Checkpoint: 487c479.

Read PLAN.md.

1. Stop the project's resident GPU load workers and verify GPU memory release.
2. Build the bounded physical matrix P/R/E exactly as frozen.
3. This must be real Qwen3-30B-A3B expert offloading:
   CPU-resident experts, real 9-MiB miss H2D, bounded GPU expert cache, real
   inter-rank activation communication and real GPU expert execution.
4. Validate Env 1=P2P/IPC and Env 2=P2P-disabled SHM in untimed preflights.
5. For every policy use separate PLAN -> COMPILE -> MEASURE -> COUNTERS phases.
   Never run controller optimization, compilation, per-event logging, profiler
   or hit/miss instrumentation inside the MEASURE region.
6. Primary eviction is Gate W128. P/R use substitution ON with frozen SERE
   calibration/thresholds. E is the substitution-OFF BR/CA control.
7. Run three clean timed repeats per policy/environment with frozen initial
   state and counterbalanced order; only the predeclared noise gate may add two
   targeted repeats.
8. Report E2E, TPOT, decode wall time and a separate counter table containing
   exact/local/substitute/effective hit, residual miss, H2D, reload, peer,
   local service and replica counters.
9. Commit phase receipts and final compact results; stop.
10. Restore resident GPU load only after all experiment processes exit.

Do not expand cache/R/batch/eviction axes, tune thresholds, run Nsight, or put
measurement instrumentation back into the timed path.
