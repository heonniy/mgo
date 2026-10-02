# PLAN — Fetch/Communication Pareto characterization on H100

Date: 2026-10-02
Status: prospective, minimum-scope go/no-go experiment.

## Goal

Before designing the final method, test whether limited expert HBM creates a measurable trade-off between:

- **Fetch cost:** CPU -> GPU expert copies / reloads.
- **Communication cost:** remote exact-expert token dispatch/combine across GPU ranks.

This packet intentionally removes substitution and load-balancing objectives.

The question is only:

> Does changing the amount of expert replication move us along a useful Fetch-vs-Comm Pareto frontier, and does the preferred point change when direct GPU P2P is disabled?

## Fixed setup

Use physical GPUs **0,1,4,5** only (R4).

- Model: Qwen3-30B-A3B-Instruct-2507 BF16.
- Local batch: **B8** only (global batch 32).
- Cache ratio: **30%**.
- One prefill + **32 decode forwards**.
- Exact experts only: **substitution disabled**.
- No quality/accuracy run is needed; require identical generated tokens across residency policies.
- Fixed eviction for this characterization: **LRU**.
- No load objective, no load oracle, no migration.
- Replicas are real GPU-resident expert copies and consume normal cache slots.
- Creating a new replica uses the existing CPU expert store -> GPU H2D path; do not add full-expert GPU-to-GPU copy.
- Keep one logical cache/residency authority.

This is not the final production policy. It is a fast physical characterization.

## Transport modes

Run the same experiment in exactly two modes.

### T0 — normal NVSwitch/NCCL

Do not force NCCL transport:

```bash
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_P2P_LEVEL
```

### T1 — P2P-disabled, SHM fallback allowed

```bash
export NCCL_P2P_DISABLE=1
unset NCCL_SHM_DISABLE
unset NCCL_P2P_LEVEL
```

Do **not** set `NCCL_SHM_DISABLE=1`. T1 is called **P2P-disabled / host-staged fallback**, not PCIe-only.

Before timing, run one short transport smoke in each mode with:

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM
```

Save the logs. Confirm that T1 does not use the direct P2P transport for the MoE communication path and that a fallback path is selected. Do not keep INFO logging enabled for primary timing.

If the runtime still selects an NVLS/direct-NVLink path for the measured MoE collective despite `NCCL_P2P_DISABLE=1`, stop and report it rather than silently calling the condition P2P-disabled.

## Stage 0A — blocked T1 transport recovery

The original `NCCL_P2P_DISABLE=1` smoke is blocked by `NET/IB/GDRDMA -> IBV_WC_RETRY_EXC_ERR` at commit `7c881f7`.

Before continuing Stage 0, execute the bounded recovery matrix in [TRANSPORT_RECOVERY.md](TRANSPORT_RECOVERY.md).

Priority is:

1. `NCCL_P2P_LEVEL=LOC` (official no-P2P cutoff);
2. if needed, additionally disable GDR with `NCCL_NET_GDR_LEVEL=LOC`;
3. if IB remains broken, disable IB and allow Socket fallback;
4. at most one final same-host `NCCL_SOCKET_IFNAME=lo` retry.

Stop at the first valid non-P2P transport. Do not load the model until this gate passes.

## Stage 0 — tiny hardware calibration

Do not run a large benchmark suite.

For T0 and T1 measure only:

1. expert H2D copy: existing **9 MiB** expert payload;
2. representative MoE peer activation transfer;
3. H2D and peer transfer issued concurrently.

Use 10 warmups + 30 timed iterations per case.

Record median and p90 only:

- H2D time / effective bandwidth;
- peer-communication time / effective bandwidth;
- concurrent slowdown of each.

Purpose: verify that disabling P2P changes the relative price of communication versus fetch.

Stop this stage after the six cells (2 modes x 3 cases).

## Stage 0B — exact-only payload capture + crossover check

The existing-trace audit at `0c09fae` proved that substituted-away raw experts do not have observed exact destination ranks, so the exact rank-pair distribution cannot be recovered from those traces without inventing a placement convention.

Before Stage 1, execute only [EXACT_PAYLOAD_CAPTURE.md](EXACT_PAYLOAD_CAPTURE.md):

1. one R4/B8 **exact-only** capture under P0 Balanced Random, substitution off, replication off, LRU, one prefill + **8 decode forwards**;
2. record the actual owner map and dispatch/combine send counts at every layer event;
3. compute exact decode payload p50/p90/p99/max;
4. run the five-size T0/R3 crossover microbenchmark;
5. stop for owner review.

No performance claim is made from the short capture. Do not implement replication or resume Stage 1 until this gate is reviewed.

If R3 is not >=1.5x slower at or below the observed exact-only p90/p99 payload range, stop the synthetic H100 P2P-off branch and move the communication-sensitive validation to the real no-NVLink server.

## Stage 0C — replica-budget Pareto screen

The exact-only payload/crossover packet at `ed7f82b` passed and showed a workload-relevant T0/R3 communication-price contrast at 32 KiB.

Before any longer trace or physical replica implementation, run only the CPU replay in [REPLICA_PARETO_SCREEN.md](REPLICA_PARETO_SCREEN.md).

Use the existing exact-only eight-decode trace and sweep:

```text
rho = {0, 0.125, 0.25, 0.50, 0.75}
```

Primary plane:

```text
x = decode peer activation bytes
y = decode expert H2D bytes
```

No GPU work is authorized in this stage.

Continue only if at least three points are nondominated and the two endpoints differ by >=10% on **both** axes.

## Stage 0D — six-cell physical F/K/C pilot

The CPU Pareto screen at `dc7b099` passed with all five budgets nondominated and selected F/K/C=rho 0/.25/.75.

Before longer traces or a final controller, execute only [PHYSICAL_FKC_PILOT.md](PHYSICAL_FKC_PILOT.md).

Freeze the CPU replay actions first, then physically execute exactly six short cells:

```text
T0-F, R3-C, T0-K, R3-K, T0-C, R3-F
```

Each cell uses the same one-prefill + eight-decode exact-only workload. No automatic repeats.

Primary question:

```text
Does the descriptively fastest rho move between T0 and R3?
```

A >=5% within-transport margin plus consistent H2D/peer mechanism counters is required to label the pilot `PROMISING_SHIFT`. Otherwise report `NO_CLEAR_SHIFT`.

Do not run Stage 1/2/3 automatically after this pilot.

## Stage 0E — real-trace communication replay, then clean F/K timing

The physical F/K/C pilot at `b6edeb0` was correctness-clean but timing-confounded by 1.87–2.87 s of replay/check CPU work and did not show a slower R3 collective interval inside the model.

Follow [TRACE_COMM_CLEAN_FK.md](TRACE_COMM_CLEAN_FK.md).

First run **communication-only actual-trace replay** using the recorded 384 decode-layer dispatch/combine matrices under T0 and R3. No model/H2D/cache work is allowed in this first gate.

Only if R3 is slower in both counter-ordered passes and the median cumulative real-trace ratio is >=1.20x may the agent proceed to a cleaned physical F/K experiment.

The clean physical stage removes heavy validation from the timed path and runs only F=rho0 and K=rho.25, two repeats each transport (8 cells total). C is not rerun.

Do not continue automatically if the communication-only gate is absent or ambiguous.

## Stage 0F — bounded P2P/CUMEM retry and conditional E1 re-run

The first E1 preflight at `b71199c` timed out after communicator initialization on the T0 `P2P/CUMEM` mapping path.

Follow [CUMEM_PREFLIGHT_RETRY.md](CUMEM_PREFLIGHT_RETRY.md):

1. run three fresh default-T0 32-KiB preflights;
2. run one diagnostic preflight with `NCCL_CUMEM_ENABLE=0`;
3. if and only if default T0 passes 3/3, commit the diagnosis and re-run the original E1 trace-communication experiment unchanged in a fresh result root;
4. if default T0 is unstable, stop after diagnosis.

The cuMem-disabled path is diagnostic only and must not silently replace T0.

## Stage 0G — rebase physical transport to stable P2P/IPC vs SHM

The bounded diagnosis at `f64eb56` found default `P2P/CUMEM` unstable (3/3 timeout) while `NCCL_CUMEM_ENABLE=0` produced a validated `P2P/IPC` direct-P2P path.

Follow [IPC_BASELINE_REBASE.md](IPC_BASELINE_REBASE.md).

New physical transport definitions:

```text
T0-IPC = NCCL_CUMEM_ENABLE=0 + direct P2P enabled -> P2P/IPC over NVSwitch
R3-SHM = NCCL_CUMEM_ENABLE=0 + P2P_LEVEL=LOC + IB_DISABLE=1 -> SHM, no direct P2P
```

The CPU replica Pareto results at `dc7b099` remain valid and must not be rerun because they depend on byte/fetch counts, not NCCL transport timing.

Run one paired smoke, then re-run the existing 384-event E1 trace communication replay under these two stable conditions. Only a STRONG_GAP may authorize the existing clean F/K physical stage.

## Stage 0H — NVLink bandwidth ladder characterization

The stable IPC E1 result at `3e59975` found only a 1.0682x median SHM/direct-P2P whole-trace gap, so model F/K remained blocked.

Follow [NVLINK_BW_LADDER.md](NVLINK_BW_LADDER.md) to characterize three communication regimes using the same frozen 384-event MoE trace:

```text
FULL-P2P = NVSwitch FULL + P2P/IPC
OFF-P2P  = NVLink bandwidth OFF + direct P2P retained (PCIe-P2P characterization)
SHM      = P2P disabled + host-staged SHM
```

The OFF mode is treated as a global server change: run only if all eight GPUs are idle, install unconditional FULL restoration, and verify FULL after restore. No sudo, job killing, waiting, or driver reset.

The CPU Pareto screen is transport-independent and must not be rerun. This stage is communication-only; it does not automatically authorize F/K model timing.

## Stage 0I — batch-size communication sensitivity

The B8 IPC/SHM replay showed large event-level latency differences but almost no whole-trace gap. Follow [BATCH_COMM_SENSITIVITY.md](BATCH_COMM_SENSITIVITY.md) to test whether this is a small-message/launch-overhead regime.

Keep cache ratio fixed at **30%** and vary only local batch:

```text
B4 / B8 / B16  (global 16 / 32 / 64)
```

Reuse the validated B8 trace; capture only missing B4 and B16 exact-only traces (1 prefill + 8 decode, R4, P0 seed42, LRU, no substitution/replication). Then replay each 384-event trace on T0-IPC and R3-SHM.

Primary metrics are whole-trace CUDA and wall time. The previous `sum_event max_rank` metric is secondary only. Also measure payload-size/fan-out distributions and a bounded 16-256 KiB latency calibration to distinguish fixed overhead from byte sensitivity.

No cache-ratio sweep or F/K model timing is authorized in this stage.

## Stage 1 — one exact-routing trace

Capture one R4/B8 exact-expert generation with substitution disabled.

The raw selected experts and token origins are the common trace for the policy sweep. Save every layer event.

Do not use this run as a performance comparison.

Because every policy executes the same exact experts, require full generated-token equality in later replays.

## Stage 2 — CPU-only replica-budget sweep

Use the captured trace first; do not spend GPU time sweeping many policies.

Fixed base behavior:

- a global miss creates at least one exact copy;
- the first copy is placed on a rank with current demand, deterministic tie break;
- remote demand can be served by an existing copy;
- optional additional replicas are placed on demanding ranks;
- every copy consumes a normal cache slot;
- LRU chooses the victim on that rank.

Sweep only the maximum fraction of global cache slots allowed to be duplicate replicas:

```text
replica_budget = {0.00, 0.125, 0.25, 0.50, 0.75}
```

For every point report over the full trace:

- physical expert H2D bytes / fetch count / reload count;
- peer activation bytes;
- remote token-rank pairs;
- mean and peak duplicate-slot fraction;
- mean unique resident experts;
- cache hit rate.

Discard dominated points in the 2-D plane:

```text
x = peer activation bytes
y = expert H2D bytes
```

The purpose is to establish a Pareto shape, not to tune a final online controller.

## Stage 3 — minimal physical validation

Choose only **three** nondominated points from Stage 2:

- **F:** lowest H2D/fetch point;
- **K:** knee/intermediate point;
- **C:** lowest communication point.

Replay the exact residency decisions physically under both T0 and T1.

Matrix:

```text
3 points x 2 transport modes x 2 repeats = 12 full generations
```

Counterbalance repeat order:

- repeat 0: F -> K -> C
- repeat 1: C -> K -> F

Alternate transport-mode order between repeats.

Do not add more repeats unless either:
- the two repeats differ by >10% in TPOT/E2E, or
- a correctness/transport check fails.

If triggered, add exactly one third repeat for the affected cells only.

## Primary metrics

For each physical cell record:

- TPOT;
- E2E generation time;
- expert H2D bytes and fetch/reload counts;
- peer activation bytes and remote-pair count;
- replica-slot fraction and unique expert coverage;
- controller time.

Use one short posthoc profile only for **K** under T0 and T1 if the primary counters show a real trade-off. Measure H2D and NCCL interval unions; do not profile all cells.

## Go / no-go interpretation

**GO** if all are true:

1. Stage 2 has at least three clearly nondominated points;
2. F and C differ materially in both directions (target: >=10% change in both H2D and peer bytes);
3. T0 and T1 assign different physical value to the same frontier, visible either in TPOT/E2E or calibrated time cost.

**NO-GO / redesign** if:

- replication changes communication but H2D/fetch barely moves;
- all useful points collapse near one endpoint;
- P2P-disabled transport does not materially change communication cost;
- runtime replication overhead dominates the intended trade-off.

Do not design a predictor, dynamic lambda, substitution policy, load objective or final Pareto selector in this packet.

## Correctness gates

Before physical timing:

1. CPU cache/replica invariants;
2. no duplicate slot ownership bugs;
3. exact expert output parity;
4. full generated-token equality across F/K/C;
5. exact H2D byte accounting for every new copy;
6. T1 NCCL transport log receipt.

## Required output

Keep it small:

- `RESULTS.md`
- `pareto_points.csv`
- `physical_runs.csv`
- `transport_calibration.csv`
- `transport_logs/` (small text logs only)
- `validation.json`

No giant Nsight traces in Git.

## Stop condition

Stop after:

- six calibration cells;
- one trace capture;
- five CPU replica-budget replays;
- twelve primary generations;
- at most two short profiles.

Return to owner review before adding any final method.
