# EXECUTION

Owner-authorized CPU-only upper-bound study.

Run only after any active physical timing job has exited; do not compete with a
timed GPU experiment for host CPU/memory resources.

Command:
```bash
cd /home/hwlee/mgo/mgo_v2
PYTHONPATH="/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:$PWD:$PWD/scripts" \
  /home/hwlee/sub-moe/phase01/.venv/bin/python -u \
  scripts/run_r8_b128_replication_stress.py
```

The script must find the completed MATH and ShareGPT 2048-request exact pools
under `/home/hwlee/mgo-results/ca_stress_workload_search_20261004/pool`.
No GPU route capture is authorized.

Search is R8 / local B128 / cache60 / Gate W128 / substitution OFF /
decode256. The selected stress triple is based on BR-only critical-rank and
peer metrics. CA and replica-oracle outputs are evaluated only after the
winner is frozen.

The temporary one-replica oracle ignores cache-capacity/creation cost and is
only a headroom bound. Do not present it as physical latency.

If neither BR nor CA free-split one-replica oracle reaches 10% reduction in
decode sum(max-rank expert rows), commit and stop. If either passes, commit and
stop for owner review; do not automatically implement persistent replication
or launch E2E timing.
