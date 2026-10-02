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
