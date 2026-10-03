# AGENT TASK — timing stability + NUMA diagnosis

Checkpoint: ef5895a.

1. Pause the current physical Env E2E/TPOT matrix immediately after any already
   active bounded run finishes. Preserve all receipts; launch no new policy cell.
2. Stop our resident GPU load workers and verify memory release.
3. Read PLAN.md and execute S0 -> S1 -> S2 -> S3 only.
4. S1 uses the exact frozen P/CA-rep/Env1 schedule truncated to decode64 and
   compares the current HEAVY PSS monitor against BOUNDARY-only monitoring in
   the prescribed six-run order.
5. Never run PSS/smaps/nvidia-smi/ps concurrently with a BOUNDARY timed region.
6. Record actual GPU/NUMA topology before choosing same-NUMA/cross-NUMA pairs.
7. S2 measures 9-MiB local-vs-remote NUMA H2D and Env2 SHM pair latency at
   32/128/512 KiB only.
8. Freeze CPU/NUMA affinity from the observed topology and run S3:
   three decode64 repeats per Env; only if stable, three decode256 confirmation
   repeats as specified.
9. Commit compact diagnostic results and stop. Do not resume BR/CA/CA-rep
   timing automatically.
