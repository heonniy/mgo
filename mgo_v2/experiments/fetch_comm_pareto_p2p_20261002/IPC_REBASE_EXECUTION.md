# IPC rebase execution receipt

Owner plan: `29e94fb`, `IPC_BASELINE_REBASE.md`.

Both conditions set NCCL_CUMEM_ENABLE=0. T0-IPC (direct P2P / NVSwitch)
clears all other transport overrides. R3-SHM (P2P disabled / host-staged)
adds only NCCL_P2P_LEVEL=LOC and NCCL_IB_DISABLE=1. NCCL INFO is confined
to one fresh 32-KiB smoke per mode. Require all four payload receipts,
one hostname/four local ranks, P2P/IPC only for T0, and SHM-only channel
paths for R3 (no required suffix). Any smoke failure stops this packet.

The trace worker's only changes are an explicit opt-in environment assertion,
CLI flag and receipt field for the owner-authorized matched cuMem setting.
Its traffic materialization, warmup, timing loop, sentinels, event intervals
and validation are unchanged. Use the existing input and aggregation/gate
conventions from TRACE_COMM_REPLAY_PROTOCOL.md: 384 events, one warmup and
three timed traces per cell, T0/R3/R3/T0 counter-order, no model/controller.

Raw output root: `/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/ipc_baseline_20261003`.
Prior raw roots remain intact. The CPU screen CSV/JSON/event CSV and frozen
schedule metadata are hash-checked before and after I0/I1; no CPU screen or
schedule generation is re-run. No F/K/C point changes.

Use GPUs 0,1,4,5 only. Retain host/GPU launch and runtime memory guards:
>=512 GiB host available and <1 GiB used per target before launch; stop at
<128 GiB host available, >32 GiB worker-tree RSS or <8 GiB free target GPU.
Bound each smoke at 90 seconds and each full replay process group at 180
seconds including startup. No automatic retries or further NCCL tuning.

Commit I0/I1 receipts and gate before any conditional clean F/K model work.
The I0/I1 launcher has no model entry point. STRONG_GAP alone authorizes
consideration of the already specified clean F/K step, with its separate
implementation/validation and two-repeat timing limits.
