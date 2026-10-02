# BATCH COMMUNICATION SENSITIVITY — when do MoE messages become bandwidth-sensitive?

Status: owner-authorized after the B8 IPC/SHM replay at `3e59975`.

## Motivation

At the existing representative point:

```text
R4, local B8 / global B32, cache30, exact-only, 8 decode
```

individual MoE communication events are clearly slower on SHM than on
NVSwitch P2P/IPC:

- dispatch p50: ~38 us -> ~62 us;
- dispatch+combine pair p50: ~74 us -> ~104-106 us.

However, whole-trace CUDA/wall time is approximately unchanged across the two
transports, while the older diagnostic
`sum_event max_rank(dispatch+combine)` gives only ~1.068x.

This suggests that the current small-message regime may be dominated by fixed
collective/launch/scheduling overhead rather than link bandwidth.

The experiment asks:

> As batch size increases and real MoE messages become larger, does transport
> sensitivity move from latency/overhead dominated toward bandwidth sensitive?

Do not change cache pressure in this packet. Cache ratio stays at 30%.

---

# Fixed system/model setup

For every new source capture:

- physical GPUs: **0,1,4,5**;
- world size: **R4**;
- model: Qwen3-30B-A3B-Instruct-2507 BF16;
- cache ratio: **30%**;
- exact experts only;
- substitution: **OFF**;
- replication: **OFF**;
- eviction: **LRU**;
- admission: existing deterministic Balanced Random P0, seed 42;
- one prefill + **8 decode forwards**;
- transport during source capture: T0-IPC;
- `NCCL_CUMEM_ENABLE=0`;
- one source capture per missing batch only.

Batch points:

```text
local B4  / global B16
local B8  / global B32
local B16 / global B64
```

The existing validated B8 source trace from `ed7f82b` / `3e59975` is reused.
Do not recapture B8 unless its prompt provenance cannot be matched.

## Prompt comparability

Use the same saved workload/prompt ordering that produced the validated B8
trace.

- B4 should be the deterministic prefix/subset of the B8 per-rank prompt set
  when the stored prompt metadata permits it.
- B16 should contain the full B8 prompt set plus the next deterministic prompts
  from the same source ordering.
- Save prompt IDs/hashes and per-rank ordering for B4/B8/B16.

If the original B8 prompt ordering cannot be recovered from existing artifacts,
do not silently create a new B8 baseline. Record `BLOCKED_PROMPT_PARITY` and
stop for owner review.

---

# Stage B0 — source trace capture for B4 and B16 only

Run exactly two new short model captures:

1. B4;
2. B16.

Each capture records for all 384 decode-layer events:

- raw selected experts;
- token-origin ranks;
- exact execution owner by expert;
- dispatch send/recv count matrices;
- combine send/recv count matrices;
- effective token routes;
- cache/plan hash;
- generated-token hash.

Correctness requirements:

- every exact selected expert has an owner;
- dispatch matrix transpose matches receive counts;
- combine matrix transpose matches receive counts;
- generated outputs are valid;
- no substitution or replication occurs;
- cache ratio remains exactly 30%.

These model captures are trace generation only. Their E2E/model times are not
performance evidence.

---

# Stage B1 — characterize communication geometry versus batch

For B4/B8/B16, report separately for dispatch and combine:

- total peer bytes per 8-decode trace;
- self bytes and self-byte fraction;
- nonzero remote message count;
- p50/p90/p99/max nonzero peer message bytes;
- mean bytes per nonzero remote message;
- per-event number of active remote ordered rank pairs;
- per-event maximum remote fan-out from a rank;
- per-event max-rank remote bytes.

Also report the union distribution for convenience.

Primary question:

```text
Does B4 -> B8 -> B16 actually increase real nonzero message size,
or mostly increase the number/fan-out of small messages?
```

Do not assume either outcome.

---

# Stage B2 — small-payload latency calibration on the stable transports

Use exactly the two already validated physical transport conditions.

## T0-IPC

```bash
export NCCL_CUMEM_ENABLE=0
unset NCCL_P2P_DISABLE
unset NCCL_P2P_LEVEL
unset NCCL_IB_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Require `P2P/IPC`.

## R3-SHM

```bash
export NCCL_CUMEM_ENABLE=0
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
export NCCL_IB_DISABLE=1
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Require SHM and no P2P/NET.

Benchmark only:

```text
16 KiB, 32 KiB, 64 KiB, 128 KiB, 256 KiB
```

per peer, with:

- 10 warmups;
- 30 timed iterations;
- max-rank median and p90;
- payload validation;
- one counter-ordered pass pair:
  T0 -> R3, then R3 -> T0.

Purpose: estimate the small-message latency floor and the point where payload
size begins to matter. This is not a bandwidth-saturation benchmark.

Report a descriptive fit over these five sizes for each transport:

```text
T(size) ~= alpha + beta * bytes
```

where `alpha` is only interpreted as a fixed-overhead proxy and `beta` as a
local byte-sensitivity proxy. Do not claim a physical link bandwidth from this
fit.

---

# Stage B3 — replay each real batch trace on T0 and R3

For each of B4/B8/B16, replay the exact captured 384-event dispatch/combine
sequence.

For B8 reuse the existing trace input, but rerun timing under the same worker
version used for B4/B16 so all three batch points share one measurement path.

For every batch use two counter-ordered passes:

```text
pass 0: T0-IPC -> R3-SHM
pass 1: R3-SHM -> T0-IPC
```

Per mode/pass:

- one untimed full-trace warmup;
- exactly three timed full traces;
- no model;
- no expert H2D;
- no cache/controller work;
- no router metadata collectives;
- all buffers allocated outside timing;
- payload validation outside timing;
- INFO disabled during timing.

## Primary timing

Unlike the earlier E1 gate, the primary metric here is:

1. **median whole-trace CUDA interval**;
2. **median whole-trace wall time**.

For each batch report:

```text
R_whole(B) = R3 whole-trace time / T0 whole-trace time
```

The old metric

```text
sum_event max_rank(dispatch+combine)
```

is retained only as a secondary diagnostic because it can choose a different
slowest rank at each event.

Also retain:

- max-rank cumulative per trace;
- event dispatch/combine/pair p50/p90/p99;
- per-repeat raw totals.

---

# Stage B4 — test the latency-bound explanation

For every real event, construct transport-independent predictors from the
frozen count matrix:

- max-rank remote bytes;
- total peer bytes;
- number of active remote ordered rank pairs;
- max rank fan-out.

Then summarize how measured event latency changes with message size.

Required diagnostics:

1. bucket events by max-rank remote bytes:
   ```text
   <=32 KiB
   32-64 KiB
   64-128 KiB
   128-256 KiB
   >256 KiB
   ```
2. report T0 and R3 event pair median/p90 in each populated bucket;
3. report R3/T0 ratio by bucket;
4. report the fraction of total events and total peer bytes in each bucket;
5. report descriptive correlation of event latency with bytes and fan-out.

Optional simple regression, if numerically stable:

```text
T_event ~= a + b * max_rank_remote_bytes + c * max_rank_fanout
```

Use it only as a diagnostic decomposition; no causal claim.

---

# Interpretation

This packet does not authorize F/K model timing automatically.

## BATCH_SENSITIVE

Report this if both hold:

- median message size or max-rank remote bytes grows materially from B4 to B16;
- whole-trace R3/T0 ratio also grows materially, target increase >=10
  percentage points from B4 to B16.

This supports the hypothesis that larger batches expose interconnect cost.

## LATENCY_DOMINATED

Report this if:

- message sizes grow but whole-trace R3/T0 stays near 1.0 (<=1.10), or
- event latency changes weakly with bytes and strongly with fixed/fan-out terms.

This supports the hypothesis that the current MoE communication regime is
dominated by per-collective/software overhead rather than byte transfer.

## MORE_MESSAGES_NOT_BIGGER_MESSAGES

Report this if larger batches mainly increase message count/fan-out while the
nonzero payload-size distribution changes little.

This would explain why byte-oriented replication may not translate cleanly
into latency savings.

## MIXED

Anything else. Report measurements without forcing one explanation.

---

# Scope / stop conditions

Maximum new model captures: **2** (B4 and B16 only).

Communication-only timing:

- B4/B8/B16 x T0/R3 x 2 pass orders;
- each cell = 1 warmup + 3 timed traces.

Do not:

- change cache ratio away from 30%;
- sweep rho;
- run F/K/C;
- enable substitution;
- add R8;
- add longer decode;
- use Nsight;
- tune NCCL channels/protocols;
- revisit NVLink bandwidth-mode controls;
- use the blocked NVLink-OFF experiment.

Commit the characterization and stop for owner review.

## Required outputs

- `batch_comm_geometry.csv/json`
- `batch_comm_trace_timing.csv/json`
- `batch_comm_event_buckets.csv`
- `small_payload_latency.csv/json`
- `BATCH_COMM_SENSITIVITY_RESULTS.md`
- `batch_comm_validation.json`
- compact prompt/trace provenance hashes.
