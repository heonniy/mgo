# Owner reduction and follow-up, 2026-10-07

Finish the current C30 B32/input256 and B32/input512 tables, all four systems,
5 primaries each with the existing common triplet selection. Completed B16 rows
remain. Cancel all C30 B64 full-baseline rows and all C60 rows (32 queued jobs).
Original running queue receives a packet-level STOP after its current worker;
this does not interrupt the active worker. The handoff then retains16 total
B16/B32 rows and resumes their remaining work.

Keep the current table's existing llama32/64 unrestricted CPU settings until
both B32 tables finish; changing it mid-table would mix baseline settings.
Then fast-forward to owner commit4943899 and rebuild the audited runner.
Run only the separate default R4/C30/B64/L512/O64 llama16/16,32/32,64/64 CPU
budget/placement audit,2 primaries each. This B64 audit is distinct from the
canceled B64 four-system matrix. All audit repeats retained, exact generated
output comparison across CPU budgets, actual override evidence and recomputed
63-step TPOT required. No C60. GPUs0,1,4,5 only.

Handoff driver: /home/hwlee/mgo-results/headline_r4_20261007/run_b32_then_llama_audit.py
Live receipt: /home/hwlee/mgo-results/headline_r4_20261007/b32_then_llama_audit_status.json
On execution/merge/parity failure preserve all evidence and report; never
silently bypass gates. Do not manually clear STOP while the handoff waits.
