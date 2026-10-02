# Batch communication execution conventions

Owner plan: `9ec43e8`; subsequent chat instruction adds local B32 (global B128).
Batches are local 4/8/16/32, R4 on GPUs 0,1,4,5, cache30, seed42/P0, LRU,
exact-only, no replication. Exactly three new source captures: B4, B16, B32.
Each is one prefill + eight decode forwards; no additional model warmup.
B8 uses the prior validated capture and byte-identical trace-count input.

Prompt provenance is frozen in `batch_comm_prompt_provenance.json`. The
saved workload matches the older checked-in SHA256. The original capture
script and raw receipts match their recorded hashes and establish rank r's
B8 indices [8r,8r+8). B4 uses its first four; B16 appends indices
[32+8r,40+8r); B32 additionally appends the next two 32-prompt blocks.
Each larger rank-local batch contains the smaller one, with no duplicate
prompt across ranks. No B8 recapture.

All conditions set CUMEM_ENABLE=0. Tiny fresh T0/R3 path smokes precede timing;
only INFO smokes must prove P2P/IPC versus SHM. No bandwidth-mode query/write.
Capture uses the same T0-IPC environment. Source captures are diagnostic,
not model timing evidence. Raw receipts stay outside Git, with hashes in Git.

The existing trace worker only gains a batch-aware input-validation argument;
its allocation, CUDA-event, communication and validation loops are unchanged.
B8 timing is rerun with that same worker as every other batch. One full
warmup plus three timed traces, two counter-ordered passes per batch.
Whole-trace CUDA and wall: take the maximum rank per repeat, then median of
three; form R3/T0 within each pass; summarize by median of two pass ratios.
The old sum_event max_rank(pair) and max_rank(sum_event pair) are secondary.
Event intervals are max across ranks for each event/repeat, pooled per cell.
CUDA intervals include launch/scheduling gaps, not isolated kernel duration.

Tiny calibration: five per-peer BF16 sizes 16/32/64/128/256 KiB, ten warmups
and thirty timed iterations, in T0,R3,R3,T0 order. Every element is checked
outside each interval; receive buffers reset to NaN. Fit alpha+beta*bytes by
ordinary least squares to five per-mode median times pooled over both passes.
Beta is a local sensitivity proxy, not physical bandwidth.

Geometry excludes self from peer bytes/nonzero remote messages; self bytes
and fractions are separately published. Phase-specific active-pair/fan-out
counts are also published. Pair predictors sum dispatch+combine outgoing
remote bytes per rank, sum their global peer bytes, and use the union of
remote directed edges for active-pair/fan-out. Bucket pair latency by this
max-rank remote-byte predictor. Fractions use unique frozen events/bytes;
latencies pool six repetitions (two passes x three) per event/mode. Correlation
uses each event's median of six intervals and Pearson correlation; constant
predictors yield null. No causal interpretation or optional regression.

Predeclared interpretation: a >=10% rise in union nonzero message median or
median max-rank pair bytes is material. BATCH_SENSITIVE requires that plus
at least 0.10 growth in the median whole-trace CUDA ratio. Report B4->B16
(original endpoint) and B4->B32 (owner extension) separately. Growing messages
with all included batch CUDA ratios <=1.10 support LATENCY_DOMINATED. If
size grows <10% but message count grows >=10%, report
MORE_MESSAGES_NOT_BIGGER_MESSAGES. Otherwise MIXED; wall ratios and variability
are always included and can qualify the interpretation. No automatic F/K.

Resource bounds: before launch host available >=512 GiB, target GPU used
<1 GiB. During execution host available >=128 GiB, target GPU free >=8 GiB,
process-tree RSS <=320 GiB for capture / <=32 GiB for communication. Timeout
600 s per capture, 180 s per communication process group. No automatic retry.

Burn handoff: eight guarded FP16 GEMM workers while no GPU experiment runs;
all eight paused for experimental stages to avoid contamination. Each allocates
three 8192x8192 matrices (384 MiB torch allocation). Resume in the stage's
finally path. PID ownership is checked before signaling; foreign jobs are never
stopped. The existing burn guard yields to foreign compute jobs/heat/memory.
Burn receipts/logs: `/home/hwlee/mgo-results/gpu_burn_batch_20261003/`.
Experiment raw root: `/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/batch_comm_20261003/`.
