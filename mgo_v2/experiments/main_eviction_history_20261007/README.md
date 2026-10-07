# Executed BR MAIN eviction-history study

Read PLAN.md, RESULTS.md and SUMMARY.json. STATUS.json is PASS.
All3 source captures and15 deterministic CPU replays completed. No run queued.

The runner is `mgo_v2/scripts/run_main_eviction_history.py`; it reuses completed
capture directories and fails if their status is not PASS. It does not launch
new captures when all3 existing directories are present. CPU-only replay:

```sh
PYTHONPATH=/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:/home/hwlee/mgo-main-table/mgo_v2:/home/hwlee/mgo-main-table/mgo_v2/scripts /home/hwlee/sub-moe/phase01/.venv/bin/python mgo_v2/scripts/main_eviction_history_replay.py --trace /home/hwlee/mgo-results/headline_r4_20261007/main_eviction_B8_L256_H256_v1/routing_trace.npz --output /tmp/main_eviction_B8_replay
```

Raw traces stay in the guarded result directories named in SUMMARY.json.
Per-batch source receipts carry hashes, all-rank agreement and exact horizon.
Step0 in CSVs is prefill; steps1..256 are decode. Cumulative CSV columns exclude
prefill. Reset/cumulative applies on eviction, not between batches or steps.
All captures were cold before prefill; policies replay the identical trace.
EXECUTED_SOURCE_HASHES.json predates the remote merge and replay-module rename.
The production default remains Near/H0/full-pinned with original decode layout.
