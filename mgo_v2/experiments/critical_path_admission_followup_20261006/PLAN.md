# PLAN — Physical critical-path attribution and admission headroom

Date: 2026-10-06  
Parent evidence: `9c20847`  
Status: prospective experiment plan. Do not claim a new policy gain from this commit.

## 1. Motivation from 9c20847

The completed R4/B128 policy-regime study invalidated packet count as a sufficient communication proxy.

Env1 C30:
- BR TPOT: 1.782411 s
- FCA TPOT: 1.935111 s (+8.57% slower)
- remote packets: 4.266M -> 3.024M (-29.1%)
- eventwise expert max: 37.852 -> 48.419 ms/step
- eventwise return max: 214.560 -> 458.070 ms/step

Env1 C60 shows the same pattern.

LA_CA avoids the severe compute skew but reduces packet count only slightly and yields <1% unsupported TPOT point gains.

Working hypothesis:

```
admission
 -> directed A2A split shape
 -> rank-local expert completion times
 -> rank arrival skew at return collective
 -> collective tail / layer completion
```

The experiment must test that chain causally before any new production controller is designed.

## 2. Frozen hardware/runtime scope

Primary hardware:
- R4 physical GPUs 0,1,4,5 on the same H100/NVSwitch server used by 9c20847.
- Primary transport environment: Env1 / normal production NCCL path from 9c20847.
- Env2 SHM is confirmation-only if a primary microbench result is ambiguous; do not automatically duplicate the full matrix.

Model-derived constants:
- hidden size = 2048;
- BF16 hidden packet bytes = 4096 B;
- fused forward packet ~= 4096 B hidden + 24 B metadata = 4120 B;
- coalesced BF16 return packet = 4096 B.

Scientific timing rules:
- no profiler in primary microbench timing;
- CUDA events for GPU duration;
- fixed GPU set and affinity;
- warmup/compile outside measured region;
- interleave condition order where practical;
- retain every valid sample;
- profiler/Nsight only in separate attribution captures.

## 3. Stage A1 — A2A split-shape microbenchmark

### Question

With total packet count and per-packet bytes controlled, does A2A latency depend strongly on the directed rank-to-rank split matrix?

### Construction

Use `torch.distributed.all_to_all_single` through the same NCCL process-group configuration as the V3 runtime.

Generate R4 directed split matrices for representative total packet counts derived from the 9c20847 BR traces. At minimum test three packet-volume levels spanning approximately the trace median to high tail.

For every total packet level construct:

1. BALANCED
   - send/recv work spread as evenly as possible over non-self peers.

2. PAIR_HOT
   - keep per-rank total send/recv close to BALANCED, but concentrate traffic on fewer source-destination edges.

3. DEST_HOT
   - multiple sources concentrate traffic onto one destination rank.

4. SRC_HOT
   - one source rank contributes a disproportionate share of outgoing traffic while preserving the same global packet count.

5. TRACE_BR and TRACE_FCA
   - replay representative real directed split matrices extracted from the C30/C60 9c20847 trajectories when available.

For each matrix measure both:
- forward packet size = 4120 B;
- return packet size = 4096 B with reversed direction.

Record:
```
total_packets
send_by_rank[r]
recv_by_rank[r]
incident_by_rank[r]
max_send
max_recv
max_incident
max_peer_edge
active_directed_edges
active_peers_by_rank
CV(send_by_rank)
CV(recv_by_rank)
```

Primary causal check:

> At matched total packets, can traffic shape alone materially change A2A completion time?

## 4. Stage A2 — Return arrival-skew microbenchmark

### Question

If payload and split matrix are identical, how much does a late rank delay return A2A completion?

Choose two fixed split matrices:
- one balanced;
- one representative 9c20847 BR/FCA matrix.

Keep all payload bytes and split counts identical.

Before return A2A, delay exactly one rank on the same CUDA stream that launches the collective. Use a calibrated GPU delay, not host sleep, so this represents late expert completion.

Target measured delays:

```
0, 0.10, 0.25, 0.50, 1.00, 2.00 ms
```

Calibrate the achieved delay with CUDA events and rotate which rank is delayed.

Measure:
- collective start/end per rank;
- rank-local NCCL residency;
- global collective completion;
- incremental tail over delta=0.

Primary relationship:

```
arrival_delta -> collective_completion_delta
```

Do not interpret NCCL residency as pure wire latency.

## 5. Stage A3 — Expert execution calibration tau(n)

### Question

Is expert row count a valid cost proxy, and what physical GPU-time function should replace it?

Use the exact compiled expert kernel and tensor shapes from the V3 runtime with one resident expert already on GPU.

Sweep:

```
n = 1,2,4,8,16,32,64,128,256,512
```

If the 9c20847 trace shows p99/max row counts above 512, extend only to the smallest power-of-two covering the trace maximum.

For every n:
- warm up outside timing;
- report median, p10, p90 GPU kernel duration;
- repeat until median relative drift is <=2%.

Then validate additivity on representative bundles of 2/4/8 experts using row-count tuples sampled from the real B128 trace.

Outputs:
- `tau(n)`;
- predicted sum tau(n_e) vs measured sequential multi-expert bundle duration;
- row-count vs GPU-time correlation.

## 6. Checkpoint A

After A1-A3, publish `MICROBENCH_RESULTS.md/json`.

Do not implement CPA yet.

The result must answer:
1. which split-matrix features predict A2A time;
2. whether arrival skew explains return tail;
3. which tau(n) predicts expert service time.

If none of these mechanisms explains the 9c20847 behavior, stop for owner review.

## 7. Stage B — Microbench-calibrated critical-path model

Build the simplest physically calibrated model supported by A1-A3.

Conceptual form:

```
T_hat_layer(x)
 = T_hat_forward(S(x))
 + max_r T_hat_expert_r(x)
 + T_hat_return(S(x), ready_times(x))
```

where:
- S(x) is the full directed rank-to-rank token packet matrix;
- T_hat_expert_r is the sum of calibrated tau(n_e) for experts executed on rank r;
- ready_times are predicted rank expert completion times.

H2D remains a separately measured term because mandatory-fetch quotas are balanced. Policy-induced future reload differences must still be reported.

### No trace-timing overfit

Derive coefficients/LUTs from A1-A3 microbench data only.

Do not fit the model to BR/FCA/LA_CA TPOT.

### Validate on existing 9c20847 captures

Use frozen C30/C60 events for:
- BR
- OLD_CA
- FCA
- LA_CA

Required validation:
- reproduce qualitative ordering: LA_CA ~= BR < OLD_CA < FCA;
- predict FCA's larger expert/return tail despite fewer total packets;
- event-level Spearman correlation >= 0.60 for the modeled critical-path quantity against measured eventwise max timing;
- median absolute relative error <=25% for the corresponding profiled phase aggregate.

If ordering fails or both numerical gates fail, the model may not drive an oracle.

## 8. Stage C — Exact current-layer critical-path oracle

Run only after Stage B passes.

At each layer:
- current routing demand is known exactly;
- resident expert owners are fixed;
- residual miss experts may be assigned to ranks;
- mandatory H2D admission-count quotas remain identical to BR;
- no replication;
- no migration;
- no future routes;
- no future cache state in the objective.

Objective:

```
min_x T_hat_layer(x)
```

using the validated Stage-B cost model.

The oracle must account for:
- directed token/rank split matrix;
- calibrated expert execution time;
- return arrival skew.

The solver may be expensive and is excluded from timed replay.

If the validated model is exactly representable with MILP-compatible auxiliary variables, use SciPy/HiGHS MILP with deterministic tie-break.

If exact representation is not possible without changing the model, stop for review. Do not label a heuristic search as an exact oracle.

### Planning and replay

1. Oracle planning pass saves event-by-event assignments and hashes.
2. Frozen-oracle physical replay injects saved assignments into a fresh run.
3. Require exact route/miss/quota/victim/cache/token parity with planning pass.
4. Report oracle solve time separately and exclude it from TPOT.

## 9. Stage D — Physical headroom gate

Primary cells:

```
R4 / B128 / C30
R4 / B128 / C60
```

Runtime:
- V3_OPT_PF_OVERLAP;
- P=2, T2;
- BF16;
- substitution OFF;
- same workload and timing boundary as 9c20847.

Compare:
- BR;
- frozen critical-path oracle.

Use the existing stable paired-repeat rule.

Report:
- TPOT/E2E;
- exposed H2D;
- forward/return A2A;
- eventwise max expert GPU time;
- remote packets and full split-matrix statistics;
- mandatory fetches/reloads;
- oracle predicted cost and actual physical gain.

### Research GO / NO-GO

Use physical TPOT, not predicted cost.

- GO: robust oracle TPOT gain >=5% in at least one cell and >=3% in the other.
- MARGINAL: 2-5% headroom.
- NO-GO for current-layer admission: <=2% in both C30 and C60.

If NO-GO, stop tuning current-layer CA/LA and pivot to multi-step cache/eviction/admission-trajectory decisions.

If GO, only then implement a low-overhead online Critical-Path-Aware Admission controller.

## 10. Required artifacts

Create in this directory:

- `MICROBENCH_MANIFEST.json`
- `A2A_SPLIT_RESULTS.csv/json`
- `ARRIVAL_SKEW_RESULTS.csv/json`
- `EXPERT_TAU_RESULTS.csv/json`
- `MICROBENCH_RESULTS.md`
- `CRITICAL_PATH_MODEL.json`
- `MODEL_VALIDATION.csv/json`
- `ORACLE_SOLVER_VALIDATION.json`
- `ORACLE_PLANNING_SUMMARY.csv/json`
- `ORACLE_PHYSICAL_REPEATS.csv/json`
- `ORACLE_PHYSICAL_SUMMARY.md/json`
- correctness/provenance/hash receipts.

Do not commit large Nsight reports.

## 11. Stop discipline

Checkpoint after A1-A3.  
Checkpoint after Stage-B model validation.  
Checkpoint after C30/C60 oracle physical replay.

Do not automatically:
- implement CPA;
- retune FCA/LA_CA;
- add lambda-weighted packet/row objectives;
- change cache ratio beyond C30/C60;
- change batch beyond B128;
- add replication/substitution;
- run R8.

Those are owner decisions after the headroom gate.
