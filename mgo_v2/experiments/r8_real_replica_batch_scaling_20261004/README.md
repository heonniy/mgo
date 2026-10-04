# R8 cache60 B128/B256 phase-aware placement and replication

This experiment searches the largest plausible multi-GPU headroom in the
high-cache/high-batch regime while preserving real cache capacity.

Common setting:
- R8;
- local batch B128 and B256;
- 60% global expert cache;
- Gate W128;
- substitution OFF;
- exact prefill + 256 decode routes.

Six policies are compared:
- BR: balanced-random mandatory-miss placement;
- CA: communication-aware balanced mandatory-miss placement;
- LA: load-aware balanced mandatory-miss placement;
- BR+REP;
- CA+REP;
- LA+REP.

All mandatory miss fetches obey the same balanced per-rank quota.

Two independent random stress workloads are selected for each batch:
- COMM-worst maximizes BR peer activation traffic;
- LOAD-worst maximizes BR critical-rank expert rows.

B128 searches sample/DP/placement seeds. B256 uses the full 2048-request pool,
so sample membership cannot change and only DP/order + placement seeds are
meaningful.

Replicas are persistent and occupy real slots. A replica may copy an already
resident expert or a just-fetched miss expert. Current-miss replication waits
for that expert's H2D, then D2D overlaps the remaining PCIe H2D queue. Expert
compute begins only after the transfer barrier.

Primary evidence remains structural counters. A calibrated CPU model uses
existing 9-MiB H2D, peer-communication and GPU expert-kernel measurements to
estimate the planned phase-separated GPU path. D2D is reported as a sensitivity
until the supplied four-GPU model-free microbenchmark is run.

The packet reports prefill and decode separately. Prefill numbers cover MoE
phases only and do not include attention/dense prefill kernels.

Run:
```bash
cd /home/hwlee/mgo/mgo_v2
PYTHONPATH="/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:$PWD:$PWD/scripts" \
  /home/hwlee/sub-moe/phase01/.venv/bin/python -u \
  scripts/run_r8_phase_aware_policy_cpu.py
```

No full-model GPU timing is automatically launched.

## Prefill interpretation

Prefill is retained as a characterization axis rather than the primary TPOT
claim. Decode replica hotness thresholds scale with local batch. For prefill,
the same multiplier is applied to the current event's per-rank token volume so
the much larger prefill token count does not trivially trigger replicas.

Report both `decode-only` replication and `all-phase` replication. The latter
captures whether a prefill-created replica helps enough to justify its slot and
its effect on the initial decode cache state. Full TTFT still requires physical
attention/dense timing and is not inferred from this CPU MoE model.
