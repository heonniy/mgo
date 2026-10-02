# Frozen E1 measurement conventions

Plan `cc7087c`, Stage E1 only. Extract the original captured P0 dispatch/combine
matrices from all four hash-verified `ed7f82b` rank receipts, events 48..431.
Do not use the later F/K/C schedules. Validate every count-matrix transpose,
8 exact experts per global token, shapes and source payload width (4096 bytes).
`trace_comm_counts.json` preserves the complete 384-event matrix sequence.

One process group per mode/pass, in T0/R3/R3/T0 order. Each runs one full
untimed warmup and exactly three timed full traces. No model import, expert
compute/H2D, cache, controller, router metadata collectives, or policy search.
One additional 32-KiB tiny preflight per mode verifies P2P/CUMEM or
SHM/direct/direct. INFO is disabled in all timed workers; no NCCL tuning.

Allocate all contiguous BF16 send/receive/expected buffers before timing.
Each event executes `torch.distributed.all_to_all_single` for dispatch then
combine, using exactly the recorded variable split sizes, including self
entries. This is the runtime's variable-size collective API. Retain original
event order. Send-buffer sentinels are exactly representable integers with
an event/phase/source/destination/element-dependent pattern; clear receives
to NaN before each trace. Validate every received element after each full
warmup/timed trace, outside CUDA timing. No event-level barrier is inserted.
One trace-boundary barrier and synchronization precede each replay.

Record CUDA events immediately before dispatch, between dispatch/combine,
and after combine. The primary per-replay total is the sum over events of
max-rank dispatch+combine intervals: sum_event max_rank pair_ms. This follows
the sequential event critical-path convention. Also publish max_rank of
cumulative event intervals and the maximum full-trace CUDA interval, so
rank-max ordering and between-event launch gaps remain visible. These are
CUDA stream intervals including API/launch waits, not isolated kernel times.
No data-dependent checks, allocations, receipts, reductions or percentile
calculations occur in the timed loop.

For a mode/pass, report the median of the three primary replay totals. Each
pass ratio is R3 median / T0 median; the gate uses the median of those two
ratios. STRONG_GAP requires both ratios strictly >1 and median >=1.20.
NO_GAP means direction reversal, or at least one ratio <=1 with median <=1.10.
Everything else is AMBIGUOUS_GAP. No rounding is applied before comparisons.
Event-level p50/p90/p99 use max-rank intervals pooled over 3 × 384 events,
separately for dispatch, combine and the paired interval; dispatch/combine
rank maxima need not be on the same rank. Quantiles use linear interpolation.

Keep source hashes, all per-rank/per-repeat/per-event timings and payload
validation receipts. Report peer bytes excluding self. The fixed trace has
139,464,704 dispatch + 301,834,240 combine peer bytes, 441,298,944 total.

Guards: target GPUs 0,1,4,5 must each have <1 GiB used and host availability
>=512 GiB before launch. Poll every 5 seconds; stop if host available <128 GiB,
process-tree RSS >32 GiB or a target GPU has <8 GiB free. Bound each lightweight
process group at 180 seconds including startup. Do not automatically retry.

Commit E1 before any conditional E2. The E1 launcher cannot start a model.
Only STRONG_GAP authorizes the separately implemented clean F/K stage.
