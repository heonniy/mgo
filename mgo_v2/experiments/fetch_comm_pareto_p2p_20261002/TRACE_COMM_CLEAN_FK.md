# TRACE-COMM REPLAY + CLEAN F/K FOLLOW-UP

Status: authorized after physical pilot commit `b6edeb0`.

## Motivation

The six-cell F/K/C pilot was correctness-clean but timing-confounded:

- T0-F vs R3-F replay/check CPU time differed by ~0.99 s;
- per-event validation/application consumed 1.87–2.87 s of the eight-step decode;
- the in-model payload-collective interval did **not** reproduce the expected slower R3 path.

Before changing the method, separate two questions:

1. Is R3 actually more expensive than T0 for the **real variable-size MoE traffic sequence**?
2. If yes, does a clean timed F/K replay show a transport-dependent preferred replica budget?

C is excluded from this follow-up because both prior physical cells were slow and its 403 GiB decode H2D endpoint is already mechanistically clear.

---

# Stage E1 — actual-trace communication-only replay

## Goal

Replay the exact 384 decode-layer communication sequence from the validated exact-only capture, with **no model, no expert compute, no expert H2D, and no cache/controller work**.

Use the recorded dispatch and combine send-count matrices from `ed7f82b`.

This stage must preserve the real per-layer rank-pair imbalance; do not replace it with equal per-peer sizes.

## Traffic replay

For every recorded decode layer event:

1. materialize BF16 buffers matching the recorded dispatch rank-pair counts;
2. execute the same variable-size all-to-all API used by the runtime;
3. materialize/execute the recorded combine traffic;
4. validate received element counts and deterministic payload sentinels outside the timed interval.

Replay all 384 events in original order.

No self traffic is counted in reported peer bytes, but runtime collectives may retain their normal self entries.

## Transport modes

### T0

Normal NVSwitch P2P:

```bash
unset NCCL_P2P_DISABLE
unset NCCL_P2P_LEVEL
unset NCCL_IB_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Expected path: `P2P/CUMEM`.

### R3

Validated non-P2P SHM:

```bash
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
export NCCL_IB_DISABLE=1
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Expected path: `SHM/direct/direct`.

Do not search additional NCCL knobs.

## Measurement

Use two counter-ordered passes:

- pass 0: T0 -> R3
- pass 1: R3 -> T0

For each mode/pass:

- one untimed full-trace warmup;
- three timed full-trace replays;
- report median of the three cumulative max-rank trace times;
- also report event-level p50/p90/p99 for dispatch, combine and dispatch+combine.

INFO logging is allowed only in one tiny preflight per mode and must be disabled for timing.

Primary ratio:

```text
trace_comm_ratio = R3 cumulative trace time / T0 cumulative trace time
```

## E1 decision

### STRONG_GAP

Proceed to Stage E2 only if:

- R3 is slower than T0 in **both** counter-ordered passes; and
- median R3/T0 cumulative trace ratio is >= **1.20x**.

### NO_GAP

If:

- R3 <= T0 in either pass with a median ratio <=1.10x, or
- the direction reverses between passes,

stop H100 synthetic P2P-off work. Record that the synthetic SHM condition does not provide a stable real-trace communication-price contrast and move future communication-sensitive validation to the real no-NVLink server.

### AMBIGUOUS_GAP

If the result lies between those gates, commit and stop for owner review. Do not launch Stage E2 automatically.

No model process is started unless E1 is STRONG_GAP.

---

# Stage E2 — clean physical F/K timing

Run this stage **only** after E1=STRONG_GAP.

## Remove harness contamination

Reuse the frozen F/K schedules from the prior pilot, but create a clean timing path.

Before any timed model run:

- fully validate schedule hashes, capacities, route destinations and expected send-count matrices;
- compile per-event actions into compact immutable arrays/records;
- validate the compact representation against the original schedules;
- run unit tests that corrupted schedules are rejected.

Inside the timed decode region, do **not** perform:

- Python set/list equality checks;
- cache hash recomputation;
- full route/send-count assertions;
- cross-rank transpose checks;
- verbose event receipts;
- policy search or greedy replica decisions.

The timed path may only:

1. read the prevalidated frozen action record;
2. apply eviction/admission/replica actions;
3. perform required CPU->GPU expert copies;
4. dispatch;
5. execute experts;
6. combine.

After the timed region, validate generated tokens, final cache state, aggregate fetch counts, aggregate peer bytes, and saved lightweight route/action receipts.

Do not subtract overhead after the fact; remove it from the timed path by construction.

## Points

Only:

- F = rho 0
- K = rho 0.25

C is not rerun.

## Runs

Eight timed model cells total:

```text
repeat 0: T0-F -> R3-K -> T0-K -> R3-F
repeat 1: R3-F -> T0-K -> R3-K -> T0-F
```

This gives two repeats for each of:

- T0-F
- T0-K
- R3-F
- R3-K

Fresh process per cell.

No third repeat automatically.

## Primary metrics

For every cell:

- eight-step decode duration;
- mean decode ms/step;
- short-run E2E;
- expert H2D fetch counts and bytes;
- peer activation bytes and remote pairs;
- minimal action-application CPU time.

The action-application CPU timer must exclude validation/checking.

If action-application time is >10% of decode time or differs by >10% between T0/R3 for the **same frozen point**, label the timing harness contaminated and stop interpretation.

Use median across the two repeats as the descriptive point estimate.

## E2 interpretation

A transport-dependent shift is supported for follow-up only if:

1. all correctness gates pass;
2. E1 established STRONG_GAP;
3. the faster of F/K differs between T0 and R3;
4. the within-transport winner beats the other point by >=5% using the two-repeat median;
5. H2D/peer counters move exactly as the frozen schedules predict;
6. cleaned application overhead passes the contamination guard above.

Otherwise report **NO_CLEAN_SHIFT**.

This follow-up still does not authorize the final weighted method, longer decode, substitution, or a paper-level speedup claim.

---

# Outputs

Stage E1:

- `trace_comm_replay.csv`
- `trace_comm_replay.json`
- `TRACE_COMM_REPLAY_RESULTS.md`
- compact T0/R3 transport receipts.

Stage E2 only if E1=STRONG_GAP:

- `clean_fk_physical.csv`
- `clean_fk_physical.json`
- `CLEAN_FK_RESULTS.md`
- validation/provenance hashes.

Commit immediately after E1. If E1 authorizes E2, commit E1 before launching model work. Commit E2 separately and stop for owner review.

## Hard stop boundaries

Do not:

- rerun C;
- add rho values;
- extend beyond 8 decode forwards;
- use Nsight;
- change NCCL channels/protocols;
- enable substitution;
- add a load objective;
- implement a weighted selector;
- start the real no-NVLink server automatically.

