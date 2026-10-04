# R8 / B128 / cache60 replication stress

This packet asks for the **maximum structural headroom** of multi-GPU expert
copy allocation in the regime where cache coverage is already high and the
global batch is large.

Frozen system:
- Qwen3-30B-A3B exact experts;
- EP ranks = 8;
- local batch = 128, global batch = 1024;
- global expert cache = 60%;
- Gate eviction, W=128;
- substitution OFF;
- decode horizon = 256.

The existing exact 2048-request pools from
`ca_stress_workload_search_20261004` are reused. No new model capture is
required.

## Stress selection

The workload is intentionally **not neutral**. We search:
1. `sample_seed`: selects 1024 requests from the 2048 pool;
2. `placement_seed`: BR one-copy random miss placement;
3. `dp_seed`: assigns 128 requests to each of the eight origin ranks.

The winner is selected using BR one-copy metrics only. CA and replica-oracle
results are hidden from selection.

Primary winner criterion:
1. maximize decode sum of per-event maximum-rank expert rows;
2. maximize BR peer bytes;
3. maximize peak maximum-rank expert rows.

This produces a reproducible stress/upper-bound workload, not a dataset-average
claim.

## Policies after freeze

On the frozen winner compare:
- BR one-copy;
- CA communication-aware one-copy;
- BR + temporary one-replica oracle;
- CA + temporary one-replica oracle.

The temporary replica oracle is deliberately an **upper bound**: it asks how
much the current event's critical rank could improve if one hot resident expert
could obtain one extra execution site without paying cache-capacity or creation
cost. It reports:
- local-origin-shift: only the target rank's own rows move to its replica;
- free-split-compute: rows may be split between original and replica to
  minimize max-rank expert rows.

No physical performance claim is made from this oracle.

If the one-replica oracle reduces decode critical-rank rows by at least 10% for
BR or CA, a second packet may implement a same-capacity persistent replica
policy with real victim/reload/D2D costs. Otherwise hot replication is stopped.

## Run

```bash
cd mgo_v2
PYTHONPATH="$PWD:$PWD/scripts" \
python scripts/run_r8_b128_replication_stress.py
```

The driver is CPU-only and reuses the existing exact route pools.
