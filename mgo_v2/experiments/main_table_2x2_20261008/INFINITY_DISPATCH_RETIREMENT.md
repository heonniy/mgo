# MoE-Infinity demand-retirement repair

The Qwen/ShareGPT B16/L1024 MoE-Infinity SDPA attempt
`mt2_qwen_sharegpt_b16_l1024_infinity_r3_v2` aborted during target repeat 2.
The native dispatcher reported `exec_state CAS failed` for expert 5 in layer 3
with state `FETCHING` (`run.log`, 2026-10-08 13:19:15 KST). The GPU resource
receipt showed only 17–21 GiB per owner GPU immediately before the abort, so
this was not an HBM OOM.

`WaitHiddenStates()` can publish the completed output before the CUDA host
retirement callback resets the expert node to `IDLE`. A subsequent token can
route to that expert while the preceding demand is still retiring. The
dispatcher previously treated this `FETCHING`/`EXECUTING` state as fatal if
`is_prefetching` was false. The companion patch makes the route worker wait
for the prior demand to retire, with a 30-second timeout that still fails a
stuck node. Prefetch and resize-reservation handling are unchanged.

The patch is applied to the local MoE-Infinity source at
`/home/hwlee/mgo-tools/headline-r4/MoE-Infinity` and the `_store` extension
was rebuilt with the existing staged build script. The previous binary is
backed up outside Git under
`/home/hwlee/mgo-results/main_table_2x2_20261008/_store.before-dispatch-retirement.so`.
The guarded four-GPU dispatch regression
`mt2_infinity_dispatch_retirement_smoke_v1` passed. Full three-repeat
measurements use new attempt labels and preserve the failed attempt; all
Qwen MoE-Infinity rows will be refreshed on this compiled version before the
final table is reported.
