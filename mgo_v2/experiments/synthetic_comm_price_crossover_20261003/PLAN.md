# PLAN — Synthetic communication-price crossover

Status: prospective, owner-authorized.

## Question

Does the preferred point on the existing fetch-vs-communication frontier move
toward more replication when only the price of remote communication increases?

Use the existing B8 decode points exactly:

| rho | fetches | H2D bytes | peer bytes |
|---:|---:|---:|---:|
| 0 | 17635 | 166424739840 | 334970880 |
| .125 | 20862 | 196878532608 | 262500352 |
| .25 | 24926 | 235231248384 | 194297856 |
| .5 | 38747 | 365662568448 | 50286592 |
| .75 | 45886 | 433034625024 | 0 |

All source files/hashes must be verified first.

## C0 — exact resource-price sweep, CPU only

Compute

```
J_byte(p, lambda) = H2D_bytes(p) + lambda * peer_bytes(p)
```

for lambda={1,2,4,8,16,32,64,128,256,512,1024,2048,4096} and solve the exact
lower-envelope crossover values.

This is a resource-price diagnostic, not latency.

From the frozen coordinates the implementation should independently recover
adjacent crossovers near 420.22, 562.34, 905.70, and 1339.76. Do not hard-code
these as results. Commit C0 before GPU work.

## C1 — freeze all five rho traces

Re-run the original CPU replay semantics and freeze the 384 decode events for
rho={0,.125,.25,.5,.75}: dispatch/combine matrices, destinations, per-rank
fetch counts, and cache hashes.

Require exact parity with the table above and send/receive transpose checks.
Do not change the old greedy replication rule.

## C2 — T0 communication-only timing

Use only the already validated T0 P2P/IPC environment. Do not rerun R3.

For every rho measure:
1. the actual 384-event communication trace;
2. a matched self-only control with the same call sequence and self rows but
   zero remote rank-pair counts.

Use two policy orders:
- pass0: 0,.125,.25,.5,.75
- pass1: .75,.5,.25,.125,0

Per cell: 1 warmup + 5 timed full traces. No model, H2D, cache/controller work
inside timing. Payload checks outside timing.

Primary:
```
T_actual = median whole-trace CUDA interval
T_self   = matched self-only median
T_remote = T_actual - T_self
```

Publish wall/event aggregations as secondary.

Time-model gate:
- T_remote must be nonnegative for rho 0/.125/.25/.5 in both passes;
- normalized remote premium should not increase by >10% as peer traffic drops;
- rho=.75 actual and self-only must agree within 10%.

If the gate fails, label TIMING_UNSTABLE and stop with the exact C0 sweep.
Do not add repetitions or tune NCCL.

Pause/restore only our model workers on GPU 0/1/4/5. Do not touch 2/3/6/7.

## C3 — H2D model from existing calibration

Do not remeasure H2D. Reuse the validated 9-MiB values:
- concurrent-4 median 0.188768 ms (primary);
- concurrent-4 p90 0.244646 ms (sensitivity).

For each event:

```
T_H2D_event = max_rank(fetch_count[event,rank]) * one_copy_cost
T_H2D_trace = sum_event T_H2D_event
```

This is a modeled critical path, not measured E2E H2D.

## C4 — synthetic time-price sweep

Only if C2 passes:

```
J_time(p,gamma) = T_H2D(p) + T_self(p) + gamma*T_remote(p)
```

gamma={1,2,4,8,16,32,64,128,256,512,1024,2048,4096}.

gamma=1 is the measured T0 remote-price baseline. gamma>1 synthetically makes
only remote communication more expensive.

Solve the lower envelope and exact positive crossover gamma values using both
the median and p90 H2D models.

Do not map gamma to a PCIe/NVLink bandwidth ratio.

## C5 — historical R3 only as context

Do not rerun R3. Mention prior R3 results only as unstable context; they are not
a calibrated gamma because prior whole-trace and microcost measurements had
strong pass/order dependence.

## Interpretation

- RESOURCE_PRICE_SHIFT: C0 moves from low rho toward high rho as lambda grows.
- TIME_PRICE_SHIFT: valid C2/C4 gives a finite gamma where nonzero rho becomes
  cheaper than rho=0.
- TIMING_UNSTABLE: C2 gate fails; keep only C0.

The key output is the first crossover scale. If it is extremely larger than
gamma=1, the mechanism may be true but practically weak for this model/cache
regime.

## Scope

B8 only. Five rho points only. No model run, no new H2D calibration, no R3,
no R8, no longer decode, no batch/cache sweep, no substitution, no new
replication policy, no physical F/K/C timing.

Outputs:
resource_price_sweep.*, frozen_rho_trace_summary.json,
comm_actual_self_timing.*, h2d_cost_model.*, time_price_sweep.* if valid,
policy_crossover.*, RESULTS.md, validation.json, one compact crossover figure.

Commit C0, then commit the bounded final result and stop.
