# Owner workload reduction: B16/L256, CPU32 only

Latest chat supersedes the B64/L51216/32/64 sweep. Use R4/C30/localB16,
global64,input256,output64, explicit CPU32/32 with fixed balanced affinity,
synchronous llama and current placement/timing checks. Disjoint warmup followed
by2 clean primaries. Only GPUs0,1,4,5. No16/64-thread follow-up.

Preserve completed B64 CPU16 repeat1 and warmup; repeat2 was owner-canceled.
B64 CPU32 was also owner-canceled after the subsequent workload change.
These partial runs must not become completed audit or main-table averages.
No full cross-thread token-parity claim is possible from the canceled sweep.

New job: llama_thread_audit_R4_C30_B16_L256_O64_t32_v1
Driver: /home/hwlee/mgo-results/headline_r4_20261007/run_llama_b16_32_only.py
Prior supervisor cleanup finishes before the new job starts. Keep all old
STOP markers and failure/cancellation receipts. Existing80-run tables retained.
