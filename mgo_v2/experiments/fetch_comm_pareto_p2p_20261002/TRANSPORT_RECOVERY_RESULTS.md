# Bounded transport recovery

Current status: **FUNCTIONAL_NO_COST_INCREASE**. Plan: `92b2070`. Original failed T1 and T0 calibration remain recorded at `7c881f7`.

Only physical GPUs 0,1,4,5 are used. Every smoke uses the same two all-to-all BF16 payloads and validates every returned row on all four ranks. The worker removes the bootstrap default `NCCL_P2P_DISABLE=0` before NCCL initialization for R conditions, leaving that variable truly unset. SHM remains enabled. No model, replica policy or system IB changes were introduced.

| Condition | Result | Direct P2P | Actual transport | Error |
|---|---|---|---|---|
| R1 | FAIL | False | NET/IB/0/GDRDMA/Shared; NET/IB/1/GDRDMA/Shared; NET/IB/4/GDRDMA/Shared; NET/IB/5/GDRDMA/Shared | IBV_WC_RETRY_EXC_ERR |
| R2 | FAIL | False | NET/IB/0/Shared; NET/IB/1/Shared; NET/IB/2/Shared; NET/IB/4/Shared; NET/IB/5/Shared; NET/IB/6/Shared | IBV_WC_RETRY_EXC_ERR |
| R3 | PASS | False | SHM/direct/direct | — |

Transport strings come from INFO channel lines; successful acceptance additionally requires payload validation on every rank and no P2P/NVLS channel. `GDRDMA` and `Shared` in a NET/IB line do not establish SHM transport.

## R1

Explicit environment: `{"NCCL_P2P_LEVEL": "LOC"}`. Other inherited `NCCL_*` variables were cleared; INFO was enabled only for smoke.

Topology lines (opaque communicator addresses are retained in the raw logs):

- `comm 0x556563975950 rank 2 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x557b5aa53b60 rank 1 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x55974ab810a0 rank 0 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x560a0039e6b0 rank 3 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`

## R2

Explicit environment: `{"NCCL_NET_GDR_C2C": "0", "NCCL_NET_GDR_LEVEL": "LOC", "NCCL_P2P_LEVEL": "LOC"}`. Other inherited `NCCL_*` variables were cleared; INFO was enabled only for smoke.

Topology lines (opaque communicator addresses are retained in the raw logs):

- `comm 0x558fef110d90 rank 3 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x55928be41d00 rank 1 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x56287be86bb0 rank 2 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`
- `comm 0x56321bfda280 rank 0 nRanks 4 nNodes 4 localRanks 1 localRank 0 MNNVL 0`

## R3

Explicit environment: `{"NCCL_IB_DISABLE": "1", "NCCL_P2P_LEVEL": "LOC"}`. Other inherited `NCCL_*` variables were cleared; INFO was enabled only for smoke.

Topology lines (opaque communicator addresses are retained in the raw logs):

- `comm 0x556346e29010 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0`
- `comm 0x55ffb4252aa0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0`
- `comm 0x562694e18310 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0`
- `comm 0x5650bf103af0 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0`

## Calibration of the first valid path

Three cases only, each with 10 warmups and 30 measured iterations. Values summarize per-iteration maximum-rank CUDA intervals; timing method and payloads match the original T0 calibration. INFO logging is off. These small messages measure software/transfer latency as well as device activity, not peak fabric bandwidth.

| Case | H2D median ms | Peer median ms | H2D p90 ms | Peer p90 ms |
|---|---:|---:|---:|---:|
| h2d | 0.198128 | — | 0.203053 | — |
| peer | — | 0.236064 | — | 0.277098 |
| concurrent | 0.191840 | 0.227200 | 0.193514 | 0.245350 |

Peer median / original T0 (0.336480 ms): **0.702x**. The original T0 was measured earlier on the shared host; this is a small calibration comparison, not an end-to-end claim.

The first functional path does not demonstrate more expensive communication on this calibration, so the required Stage 1 cost-increase gate is not met. The preferred 2x contrast is also absent. Stage 1 has not started. The smaller measured median does not establish that SHM is intrinsically faster: the runs were separated in time on a shared host. Do not try later R conditions or retune messages merely to obtain a larger ratio.

The original `artifact_hashes.json` is the historical Stage 0 snapshot; owner edits at `92b2070` and this recovery are separate. Current recovery artifacts/source hashes are in `transport_recovery_hashes.json`. Raw trial commands, environment, source hashes and memory samples are retained alongside each smoke.
