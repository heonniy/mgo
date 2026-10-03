# f317328 execution freeze

S1's live worker, order, affinity, schedule and measurements remain unchanged.
After all six accepted runs and the S1 summary, the old S2 subprocess entry
hands off to the amended driver. It verifies all six PASS receipts, retires
only the now-waiting original coordinator (no active science child), and starts
the replacement S2A/S2B/S2C/S3 sequence in its own process session. The old S2
local/remote labels and old S3 continuation are never launched.

## S2A semantics and prospective decision rule

NORMAL preserves existing behavior: full file-backed CPU pool touch during
load, full untimed warmup, then pre-generation mincore residency checks.
PRETOUCH adds a CPU read of every 4-KiB page of each expert this rank's frozen
schedule can fetch, immediately before timing. It does not copy to GPU or
change any cache key or action. The experiment does not manufacture cold cache,
drop system cache, or demonstrate a cold-versus-warm disk comparison.

Order is NORMAL, PRETOUCH, PRETOUCH, NORMAL; decode64, Env1, original node-wide
CPU affinity, BOUNDARY monitoring. No GPU test is added before these four runs.
The actual model loop is AST-identical to S1. Before/after rank-process
getrusage faults and host MemAvailable/Cached are collected outside globally
synchronized timing boundaries. These fault counts include boundary barriers
but exclude warmup/PRETOUCH and do not isolate expert pages from other rank
allocations. File backing or minor faults alone are not evidence of disk I/O.

To operationalize the owner's word "materially" before seeing S2A outcomes,
use this conservative fixed rule: NORMAL decode spread >5%, PRETOUCH spread
<=5% and <=half NORMAL spread, PRETOUCH median no worse, and >=50% reduction in
a nonzero median minor or major fault count without an increase in the other
category. Only then mark PAGE_RESIDENCY_CONFOUND and use PRETOUCH identically
for both S3 environments. Otherwise S3 remains NORMAL. Two samples per mode
are descriptive evidence; the label is not expert-specific causal proof.

## S2B and S2C

H2D uses nested GPU-ID sets {0}, {0,1}, {0,1,2,3}, {0..7}, one pinned 9-MiB
CPU buffer per rank, 20 warmups and 50 measured copies per level. CPU barriers
coordinate starts outside timing, and CUDA events measure per-GPU copy time.
Aggregate bandwidth uses total bytes over latest host completion minus earliest
host start; start skew is retained. Per-GPU ratios reference the single-GPU0
baseline, not an unmeasured individual baseline for each other GPU. No physical
NUMA or PCIe-layout inference follows from the chosen IDs.

SHM uses an untimed eight-rank Env2 preflight followed by a fresh eight-rank
process with NCCL debug disabled. All 28 unordered pairs are measured sequentially
at 32/128/512 KiB, each with 20 warmups and 50 measured dispatch-return iterations.
Inactive ranks wait on a CPU group, not GPU work. Pair order is lexicographic,
low ID initiates; all per-rank samples are retained and initiator round trips
form the 84-row matrix. Report max/min median pair ratio per payload. Pair labels
are GPU IDs only. Disjoint fixed CPU affinity applies throughout these probes.
No heavy or live resource monitor executes during model-free measurements.

## S3 and handoff

S3 retains the prescribed Env1,Env2,Env2,Env1,Env1,Env2 order, disjoint guest
CPU sets and <=5% decode spread per environment. Only a passing decode64 gate
permits exactly three decode256 confirmations in each environment. Failure
stops the packet; no policy matrix resumes automatically. Resource/STOP checks
remain at boundaries. Resident model load resumes only after scientific workers
exit, subject to its existing resource/foreign-process guards.

H6 physical host NUMA remains UNKNOWN. Guest topology is not host topology.
