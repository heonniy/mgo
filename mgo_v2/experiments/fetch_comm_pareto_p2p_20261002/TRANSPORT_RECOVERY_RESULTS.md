# Bounded transport recovery

Current status: **RECOVERY_IN_PROGRESS**. Plan: `92b2070`. Original failed T1 and T0 calibration remain recorded at `7c881f7`.

Only physical GPUs 0,1,4,5 are used. Every smoke uses the same two all-to-all BF16 payloads and validates every returned row on all four ranks. The worker removes the bootstrap default `NCCL_P2P_DISABLE=0` before NCCL initialization for R conditions, leaving that variable truly unset. SHM remains enabled. No model, replica policy or system IB changes were introduced.

| Condition | Result | Direct P2P | Actual transport | Error |
|---|---|---|---|---|
| R1 | FAIL | False | NET/IB/0/GDRDMA/Shared; NET/IB/1/GDRDMA/Shared; NET/IB/4/GDRDMA/Shared; NET/IB/5/GDRDMA/Shared | IBV_WC_RETRY_EXC_ERR |

Transport strings come from INFO channel lines; successful acceptance additionally requires payload validation on every rank and no P2P/NVLS channel. `GDRDMA` and `Shared` in a NET/IB line do not establish SHM transport.

## R1

Explicit environment: `{"NCCL_P2P_LEVEL": "LOC"}`. Other inherited `NCCL_*` variables were cleared; INFO was enabled only for smoke.

Topology lines (opaque communicator addresses are retained in the raw logs):

- `comm 0x556563975950 rank 2 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x557b5aa53b60 rank 1 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x55974ab810a0 rank 0 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x560a0039e6b0 rank 3 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`

The original `artifact_hashes.json` is the historical Stage 0 snapshot; owner edits at `92b2070` and this recovery are separate. Current recovery artifacts/source hashes are in `transport_recovery_hashes.json`. Raw trial commands, environment, source hashes and memory samples are retained alongside each smoke.
