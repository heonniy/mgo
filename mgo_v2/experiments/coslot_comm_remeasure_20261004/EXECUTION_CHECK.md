# Execution prerequisite check

The owner replaced the running OldCA experiment with commit f778cf9 on this
branch. OldCA R8 was cancelled and its interrupted timing excluded; completed
R4 receipts remain on the previous branch at cancellation commit d45e624.

Execution is blocked: the configured raw root has no R_BR_env1_PLAN_* or
R_CA_env1_PLAN_* directories. Its validated schedules are P-cell schedules.
Other stress-study R4 schedules have different inputs and are not substitutes.
The existing raw-root STOP marker is preserved. No new PLAN or GPU run started.
The owner was asked for the missing PLAN location or a cell change.

Prepared corrections before execution:
- CoSLoT returns accumulate in the original ascending expert order, avoiding a
  BF16 rounding change caused by destination-major transport order.
- CoSLoT COUNTERS consumes frozen schedules and original logical metrics;
  it does not rerun the placement controller or global-route collectives.
- CoSLoT measurements use fixed 24-CPU affinity per rank and a ready/GO
  boundary. Resource scans stop during MEASURE and resume after ranks exit.
  Full warmup, model inputs, cache actions and three repeats remain unchanged.

Three CPU layout tests pass, including a rounding-sensitive accumulation-order
fixture. Python syntax checks pass. GPU correctness/performance remain untested
until the required frozen PLANs are available.
