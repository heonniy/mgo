# R4 B64 main_OURS EP overhead diagnosis

Use the frozen Qwen3 ShareGPT main-table cell
`Qwen3_ShareGPT_R4_C30_B64_L512_O64`: physical GPUs 0/1/4/5, local B64,
input 512, output 64 and C30 expert slots 461/461/461/460. Run the current
main_OURS default: C++ Ready-First expert executor, Near placement, optimized
prefill/decode layout, rank-private pinned CPU expert source and prefetch OFF.
Compare three code paths on exactly that workload: (1) pre-optimization
commit `adc8f44d86c9a8f162d4fb896f826ad770b9d0bf` from an isolated
worktree, (2) current selected main_OURS with compiled gate-history and bulk
layout views but without `--compiled-dense`, and (3) the same runtime with the
Qwen-only `--compiled-dense` candidate. None changes the C++ expert executor,
Near placement, cache capacity, communication collectives or routing seed.
The default path in (2) is the meaning of “general EP” in this packet. Do not
substitute a static-owner or other-system baseline.

Run two clean target repetitions per code path after a disjoint warmup and
cache reset. Record global TTFT, TPOT, E2E and rank-level validation. Then
run one separate diagnostic job per path with the same frozen target, a
full-decode profile and an independent one-token prefill profile. Require
exact target tokens, final cache-state hash and H2D-byte parity. Use the
existing eight-load guard so
all owned background model processes are quiet during timing and restored
after each job. Preserve raw request manifests outside Git.

Report TTFT and TPOT phase partitions separately, including routing metadata,
controller/layout, demand H2D, forward packet/collective, expert compute,
return partial/collective and unclassified runtime. Diagnose max-rank
critical-path effects and distinguish H2D copy-stream service from exposed
wait. CUDA current-stream spans include host launch gaps and peer arrival
waits; they are not pure NCCL wire or Python times. Do not add overlapping
H2D service to the 100% partition. For each rank report MAIN expert-use/hit/
miss counts, token-expert rows, H2D copies/bytes, per-copy PCIe service time
and the total copy-stream service. Report the maximum exposed H2D wait on the
current-stream critical path separately: total DMA service is not TPOT
latency because it may overlap compute or communication. Use uninstrumented
runs for primary TTFT
and TPOT, and do not call the diagnostic elapsed time a primary performance
number.

## Strict hit-then-miss grouped executor follow-up

On the current selected main_OURS code, keep the same frozen cell and compare
two expert execution schedules with two uninstrumented repetitions each:

- A: C++ Ready-First, individual expert GEMMs (the main_OURS default and the
  current-code baseline above).
- N (`new_OURS`): execute **only resident cache hits** in one grouped GEMM
  wave; after that wave is submitted, wait for all current demand-miss H2D
  transfers to complete and execute **all misses** in a second grouped GEMM
  wave (`hit_then_miss`). An already-ready miss still belongs to wave two.
  If there are no hits or misses, omit the corresponding empty wave.

The previously planned B (`serial_all`) run is retained as a secondary
reference because it started before the strict user clarification. Do not
conflate B or the older C (`two_wave`, which admits ready misses in wave one)
with `new_OURS`. The primary comparison is A versus N on the same code
revision, routing, cache and workload, requiring token/cache/H2D parity.
Use a separate diagnostic pass for each primary arm to attribute differences
and report first/second-wave hit/miss group counts. Keep the pre/post metadata
and index optimization comparison distinct from the A/N scheduling comparison.
