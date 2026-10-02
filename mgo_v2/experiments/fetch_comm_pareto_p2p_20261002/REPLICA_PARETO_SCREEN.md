# REPLICA PARETO SCREEN — CPU-only go/no-go

Status: authorized after `ed7f82b`.

## Goal

Use the validated exact-only R4/B8 capture from `eb65df0` / `ed7f82b` to answer one question before any more GPU work:

> Does allowing duplicate expert residency create a real trade-off between CPU expert H2D/fetch and inter-GPU activation communication?

This is a **trace replay characterization**, not the final placement method.

No new model generation, NCCL run, quality evaluation or GPU timing is allowed in this stage.

## Input

Reuse the exact-only capture:

- R4, physical GPUs 0,1,4,5;
- B8 per rank / global batch 32;
- cache ratio 30%;
- substitution off;
- replication off in the captured execution;
- LRU;
- one prefill + 8 decode forwards;
- raw exact routes, token origins, owner/send-count validation already PASS.

Replay the same raw exact route sequence from an empty cache. Policy-dependent residency may differ; raw routing is frozen.

## Replica budget

Sweep exactly:

```text
rho = {0.00, 0.125, 0.25, 0.50, 0.75}
```

Let:

```text
duplicate_slots = total_resident_copies - unique_resident_experts
```

and require:

```text
duplicate_slots <= floor(rho * global_cache_slots)
```

Per-rank physical slot capacities remain unchanged.

## Deterministic replay policy

Keep this simple and identical across all rho except for the replica cap.

### 1. Exact routing only

No substitution.

Every raw selected expert must be served exactly.

### 2. First copy on global miss

If expert e has no resident copy:

- count current token-route demand for e by origin rank;
- place the first copy on the rank with largest current demand;
- deterministic tie break: smallest rank;
- if full, evict that rank's LRU legal victim.

This first-copy rule is the same for every rho.

### 3. Serving an already resident expert

For a token originating on rank r:

- use a local copy of e if one exists on r;
- otherwise use e's deterministic primary copy;
- primary is the original first copy while resident;
- if the primary is evicted but another replica survives, the smallest-rank surviving copy becomes primary.

No topology-aware remote-owner choice is added in this screen.

### 4. Optional on-demand replica

After mandatory first-copy admissions are resolved but before dispatch, consider a replica (e,r) only when:

- e is active in the current event;
- rank r has current demand for e;
- e is resident globally but not on r;
- the global duplicate-slot cap permits one more replica.

A replica is loaded from the CPU expert store through the same H2D path as any other copy.

For every legal candidate, compute **exact current-event peer activation bytes saved** by adding the local copy and recomputing dispatch/combine rank-pair counts.

Greedily select the candidate with the largest positive byte saving; tie break by expert id then rank. Repeat until:

- no positive-saving candidate remains;
- replica budget is full; or
- no legal victim exists.

This greedy rule is only a controlled way to populate the budget. It is not the final method.

### 5. Victim rule

Use LRU only.

During an event:

- never evict a copy of any expert active in that event;
- never partially modify state if no legal victim exists;
- replica creation may evict an inactive **unique** expert, intentionally exposing the coverage-vs-locality trade-off;
- if an evicted expert still has another copy, it remains globally resident.

## Accounting

Every new physical copy from CPU counts as one H2D fetch, including:

- first copies on global misses;
- reloads after all copies were previously evicted;
- additional replicas.

Report these separately:

```text
first_copy_fetches
reload_fetches
replica_fetches
total_fetches
total_H2D_bytes
```

Use the bound 9 MiB physical expert size for H2D byte accounting.

Compute exact peer activation bytes from the replayed owner/copy state using the same dispatch/combine semantics validated by the exact capture.

Primary Pareto plane:

```text
x = total peer activation bytes
y = total expert H2D bytes
```

Report both full-trace and decode-only totals. The **decode-only plane is primary**; prefill is retained as context because it initializes cache state.

Also report:

- remote token-rank pairs;
- mean/peak duplicate-slot fraction;
- mean unique resident experts;
- local exact-hit fraction;
- global-resident remote-service fraction;
- cache miss/reload counts.

## Validation

Before accepting the sweep:

1. rho=0 has zero duplicate slots at every event;
2. every route has exactly one execution destination;
3. no rank exceeds slot capacity;
4. duplicate cap is never exceeded;
5. LRU/tie breaks are deterministic;
6. send/receive matrices transpose-match;
7. rho=0 replay is internally checked against an independent implementation of the same deterministic first-copy rule.

Do **not** require rho=0 to reproduce the captured P0 owner trajectory; the screen deliberately uses one common deterministic first-copy rule for all budgets.

## Pareto analysis

For the five rho points:

- remove dominated points;
- publish all raw points anyway;
- identify:
  - F = minimum decode H2D bytes;
  - C = minimum decode peer bytes;
  - K = nondominated knee if one exists.

Do not invent K if there are fewer than three nondominated points.

## Go / no-go

**GO to a longer trace / physical F-K-C validation** only if:

1. at least three rho settings are nondominated; and
2. F vs C changes decode H2D bytes by >=10%; and
3. F vs C changes decode peer activation bytes by >=10%.

If only two useful endpoints exist, report that shape and stop for owner review.

If replication materially changes only one axis, stop and reconsider the research framing.

## Outputs

Keep this stage CPU-only and compact:

- `replica_pareto_screen.csv`
- `replica_pareto_events.csv` only if reasonably small; otherwise summary JSON only;
- `replica_pareto_screen.json`
- `REPLICA_PARETO_RESULTS.md`
- validation/provenance hashes.

Commit immediately when complete and stop. Do not run Stage 3 automatically.
