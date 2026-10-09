# Complete seven-policy live cohort after best64 search

The owner-selected TPOT improvements apply to every arm: grouped expert GEMM
for decode, native C++ gate history/metadata and controller, and the existing
native prefill path. Prior native R/G results and the interrupted native
Stage-II warmup remain separate historical evidence. This amendment is frozen
before any new Stage-II primary and changes no quota formula, assignment
objective, cache budget, input, output length or expert weights.

Run R-NEAR, G-NEAR, G-BR, G-CA, G-NUMA-CA, R-BR and R-CA in one process cohort
on physical GPUs0,1,4,5 with NUMA groups[0,1] and[4,5]. Use the original frozen
headline64 target and disjoint warmup, not the selected best64 input. Preserve
local16/input512/output64/C30, MAIN[459,459,459,458]+P2x4, seed42, greedy with
no EOS stopping, prefetch/substitution/replication OFF. Reuse only model and
NUMA-shared source; cold cache, controller, metadata/history and KV state are
reset for every generation. Two ranks share each54GiB fully pinned source;
verify inodes/page locality and108GiB unique source size. Maintain128GiB host
headroom and the exclusive GPU lease; stop owned burn before GPU use and
restore it after all owned workers exit.

The exact globally serialized order is metadata/PLAN completion, forward A2A
completion, mandatory H2D completion, expert compute completion and return A2A
completion. Rendezvous use Gloo and no H2D overlaps compute or NCCL. C++ Near,
BR and CA fill their supplied hard quotas. NUMA-CA uses the existing frozen
measured peer-cost matrix; no tuning from live timing outcomes.

Every arm receives a full64 warmup with grouped weighted-output relative
L2<=1%, exact wire/layout checks and a separate one-token prefill validation.
Primary schedules are all seven arms forward, reverse, forward (three repeats).
If any arm's (max-min)/median TPOT exceeds5%, add reverse/forward repetitions
for ALL seven arms. Use the existing10800s job bound and3600s per-generation
bound. Retain all attempts and never remove a slow repeat. Reject compilation,
nonfinite logits, changed per-arm token/cache/full61-column trace hashes, or
cache/phase violations; preserve failure receipts and repair the cause.

After primaries, collect one separate timers-only diagnostic per arm, including
actual mandatory expert assignments and all-rank phase intervals. Assignment
copies occur only in diagnostics, whose PLAN residual includes their overhead.
Do not capture large hypothetical fixed-miss snapshots inside these live
diagnostics; the previously completed conditional fixed-snapshot replay remains
the evidence for identical miss sets/quotas. Independent live greedy policies
can change future M, quotas, routing and cache trajectories. These live results
must not be presented as a same-miss causal placement experiment.

Report all TTFT/TPOT/E2E repeats and median/mean/SD/range, per-GPU HBM and
post-generation RSS, unique pinned memory, per-event quota/actual assignments,
first/reload fetches and evictions, H2D and peer bytes, projected/actual expert
work skew, complete MoE phase breakdown and output agreement/divergence.
Diagnostic spans are excluded from primary statistics. Validate3072 events
per arm on all four ranks and compare diagnostic token/cache/full policy trace
to every unprofiled repetition. Commit each validated arm separately, followed
by the common comparison report. Keep data manifests under data2 and store
their hashes/path receipts in Git.

Execute only after the best64 pipeline terminates and releases its lease:

```sh
/data2/esjung/envs/mgo-pcie/bin/python -u mgo_v2/scripts/run_pcie_ours_job.py \
  --arm R-NEAR --sequence stage2_grouped --timeout 10800 \
  --out /data2/esjung/mgo-results/pcie_topology_ablation_20261009/G4_stage2_grouped_attempt1
```

This seven-policy cohort is prepared, NOT RUN. External baselines follow its
correctness/measurement/commit gates; the complete original experiment goal
remains active.
