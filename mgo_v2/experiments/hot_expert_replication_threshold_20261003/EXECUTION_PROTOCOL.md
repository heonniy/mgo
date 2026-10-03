# Frozen execution conventions

The single-copy placement packet completed and committed at `85283b0` before
this packet starts. Preserve the owner's shutdown of GPU 2/3/6/7. Only the
existing model workers on GPU 0/1/4/5 may be paused for H1 and restored.

H0 reuses the hash-verified archived exact F event states from the completed
rank-local study. Verify raw receipt sizes/hashes and archive hashes again.
Report positive current demand occurrences, remote-only occurrences, and
per-pair concentration ranked by cumulative remote demand across eight steps.
Individual exact marginal peer savings are nonadditive due to dispatch sharing.
The fraction denominator is explicitly their sum, not physical wire traffic.
CPU stages use one process, threads=1, hidden CUDA and hard 4-GiB address space.

H1 uses the established cuMem-disabled T0-IPC and R3-SHM settings without
additional tuning. One bounded smoke per mode verifies transport with INFO
logging; timing processes omit INFO logging. Pair origin/owner are logical
ranks 0/1 (physical GPU 0/1), with a four-rank communicator on 0/1/4/5.
Use actual NCCL send/receive for dispatch followed by return of the received
BF16 rows. All rank intervals are retained; summarize paired max-rank samples.
Run transport order T0,R3 then R3,T0, exactly 10 warmups and 30 timed pairs
per size per pass. Derive thresholds from the pooled 60 max-rank samples per
size/mode; retain each pass's median/p90 and crossover to expose order effects.

H1 H2D measures one contiguous pinned BF16 expert-shaped (3,2048,768) copy,
exactly 9 MiB, with the existing nonblocking pinned-copy path. Measure only
rank0 active, then all four ranks concurrently; 10 warmups and 30 samples per
condition. Release default streams from a common GPU barrier before copying;
no CPU synchronization is inserted between barrier and copy. Payload checks
occur outside all measured intervals. These are isolated transport costs,
not full model or controller timing.

H2 reports all transport x H2D-condition x median/p90 thresholds. A missing
measured crossover is `>512` without extrapolation. Because H3 is capped at
four thresholds per batch, use the four-rank concurrent H2D condition for
H3 (T0/R3 x median/p90), deduplicating identical measured thresholds. This
choice was proposed to the owner before measurement; absent a preference,
use the proposed concurrent condition to represent simultaneous fetch pressure.
H4 uses the optimistic single-rank R3 median threshold exactly as planned.

H3 runs only batch/threshold cells with nonzero F remote candidates. Common
F prefill, then greedy current exact marginal saving among n>=threshold
candidates, deterministic expert/rank ties, original physical slots/LRU and
active protection, no separate replica quota. No future score or predictor.
Reuse existing lifecycle and victim-induced reload accounting. Historical
K/C prefill differs; report actual byte coordinates without a latency claim.
