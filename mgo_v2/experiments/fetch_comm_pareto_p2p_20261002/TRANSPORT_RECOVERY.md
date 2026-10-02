# TRANSPORT RECOVERY — unblock the P2P-disabled H100 condition

Status: authorized next step after commit `7c881f7`.

Observed failure:

- T0 completes with `P2P/CUMEM`.
- `NCCL_P2P_DISABLE=1` is honored.
- NCCL then selects `NET/IB/GDRDMA` instead of SHM.
- The first all-to-all fails with `IBV_WC_RETRY_EXC_ERR`.
- No model/replica experiment may start until a functional non-P2P transport smoke passes.

The purpose here is not to repair the cluster's InfiniBand fabric. It is to find the smallest reproducible **non-P2P** transport condition for the Fetch-vs-Comm experiment. Run only the tiny all-to-all smoke from Stage 0. Stop at the first valid condition.

## R1 — prefer P2P_LEVEL=LOC instead of P2P_DISABLE

NCCL documents `NCCL_P2P_LEVEL=LOC` as "never use P2P".

Run:

```bash
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
unset NCCL_IB_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Enable INFO logging for this smoke only.

Acceptance:

- no channel/path reports direct `P2P`;
- all-to-all payload validation passes;
- record the actually selected transport.

If R1 succeeds, use **R1 as T1** and do not run R2/R3.

## R2 — keep IB but disable GPU Direct RDMA

Only if R1 again selects NET/IB and fails.

Run:

```bash
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_NET_GDR_C2C=0
unset NCCL_IB_DISABLE
unset NCCL_SHM_DISABLE
```

This removes direct GPU-to-NIC RDMA while leaving the IB transport available.

Acceptance is the same: no direct P2P, smoke passes, actual transport recorded.

If R2 succeeds, label the condition explicitly:

`P2P-disabled + GDR-disabled NET/IB`

Do not call it SHM or PCIe-only.

## R3 — bypass the broken IB path and use Socket

Only if R2 still fails with an IB completion/connection error.

Run first:

```bash
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
export NCCL_IB_DISABLE=1
unset NCCL_SHM_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
```

If NCCL chooses Socket but selects an external interface and the same-host smoke fails, do one final retry with:

```bash
export NCCL_SOCKET_IFNAME=lo
```

All ranks are on the same host; this final retry is a synthetic same-host host/network-stack condition.

If R3 succeeds, label it:

`P2P-disabled + IB-disabled Socket`

This is a **synthetic slow-communication stress condition**, not a PCIe-only server.

## Do not spend time on

Do not:

- modify system IB configuration;
- load/unload kernel modules;
- use sudo;
- retune NCCL performance variables;
- run the model between failed transport smokes;
- run more than R1/R2/R3 plus the single loopback retry;
- force `NCCL_SHM_DISABLE=1`.

The existing log reports an unusual `nNodes 4 localRanks 1` interpretation under failed T1. Preserve this line in the recovery result, but do not turn this packet into a long NCCL topology investigation.

## Success gate

As soon as one R-condition passes:

1. commit the successful smoke and logs;
2. run the original three tiny calibration cells under that exact environment;
3. compare communication median against T0;
4. proceed to Stage 1 only if communication is measurably more expensive.

Prefer a condition with peer median >= 2x T0. If the first functional non-P2P path is <2x T0, report it before continuing.

## Failure gate

If all bounded recovery conditions fail:

- commit `BLOCKED_SYNTHETIC_TRANSPORT`;
- stop H100 P2P-off work;
- continue the Pareto characterization on the real no-NVLink/PCIe multi-GPU server instead.

Do not keep searching NCCL knobs after this bounded recovery matrix.

## Artifacts

Commit immediately after each attempted recovery condition:

- `transport_recovery.csv`
- one small INFO log set per condition;
- `transport_recovery_result.json`

Suggested CSV columns:

```text
condition,status,p2p_used,selected_transport,error,peer_median_ms,h2d_median_ms,notes
```
