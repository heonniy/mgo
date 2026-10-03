# In progress — no accepted timing results yet

- R8/R4 Env 1 P2P/IPC and Env 2 SHM/direct/direct preflights: PASS.
- Incremental live policy versus the CPU reference: PASS, including full
  256-decode replica-admission checks on P and R source traces.
- Frozen communication layout ordering: PASS in 60 explicit-reference cases.
- First P/BR physical PLAN is in progress; PLAN/COMPILE validation must pass
  before any MEASURE is accepted.
- Initial preparation failures are retained in `phase_receipts/`: conservative
  RSS guard stops (no OOM) and a corrected generation-config initialization
  error. Shared-memory accounting now uses PSS plus host-available guards.

Raw logs and guarded process state:
`/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003/`.
Resident idle-load workers remain stopped while the physical study is active.
See EXECUTION.md for implementation, safety and interpretation conventions.
