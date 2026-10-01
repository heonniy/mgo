# Current-server validation — 2026-10-01 UTC

Base: `codex/mgo-v2-runtime-20261001` at
`1ded6bbf5058716db52b2299cd9b50ec50b6301e`.
Raw evidence: `/home/hwlee/mgo-results/runtime_validation_20261001` on
`cloud-0n58xq`. The final measurement section is updated after the matrix
finishes; correctness and profiling below are completed results.

## Environment

- Eight H100 80GB GPUs, NV18 connectivity between every pair, CUDA 12.8.
- One process exposes one GPU. The OS exposes only NUMA node 0; PCI sysfs
  reports -1. Strict CPU/memory binding is applied to the sole OS node and
  verified with `get_mempolicy`. Physical locality beyond the VM's exposed
  topology cannot be inferred.
- PyTorch 2.11.0+cu128, Transformers 4.57.6, CUTLASS 3.5.1.
- Checkpoint: `/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507`, BF16,
  48 layers, 128 experts/layer, top-8, hidden 2048, expert intermediate 768.
- Dense layers and attention run on GPU. No RPC expert path, legacy Python
  residency controller, migration or GPU expert replication is enabled.
- The overlapping `hard_affinity_gpu_validation_20261001` launcher and its
  eight workers were stopped under the user's authorization. The exact
  process receipt is `stopped_experiments.json`; existing results were kept.

## Defects fixed

1. The smoke loader entered the legacy Python model/control stack. A new
   checkpoint loader creates native HF dense GPU modules and directly loads
   only the C++ host-store/slot extension. Experts remain meta placeholders
   in HF and execute through the single `GlobalExpertController`.
2. Caller, worker and return CUDA streams lacked sufficient dependencies.
   Input/output events now cover routing snapshots, gathered inputs and
   asynchronously stored partials, including non-default caller streams.
3. Fused BF16 arithmetic and rank-first accumulation changed native Qwen3
   rounding. One diagnostic run produced 429 router-choice mismatches despite
   matching generated tokens. The default now uses native projection/SiLU
   rounding and origin-side accumulation in expert order. The return protocol
   sends real weighted expert vectors without zero-padded top-k payloads.
4. Fetching into a physical slot did not rebind the tensor index used by the
   computing module. Its old CPU views caused additional host-weight reads.
   Slot commits now install actual CUDA tensor views. Debug audits check
   addresses as well as owner/expert/slot maps. This was found by the new
   direct-view assertion and independently confirmed in the old Nsight trace.
5. Tied Coverage percentiles were assigned different ranks, and normalized
   floating-point scores could split exact ties by one ULP. Integer doubled
   midranks preserve the validated tie-break. Float32 similarity=.65 is
   inclusive; gate history matches the trace packer's mean rounding.
6. Impossible hard quotas now fail before changing cache residency, and
   affinity-based policies reject missing, malformed or nonfinite tables.
7. Resident weights are read directly from pinned execution slots. A drained,
   explicit experiment reset can resize the physical pool together with a
   fresh controller. Hungarian repeated-rank costs, exact swap deltas, bulk
   gate-history updates and token packet construction avoid redundant work.

8. Left-padded batched generation now supplies per-sequence position IDs.
   Padding rows are excluded from expert execution, policy history and traffic.
   The original dense attention mask still controls attention. CPU regressions
   cover decode refresh, empty valid inputs and detaching the root mask hook.
9. Coverage losses are updated only for changed resident layers. A randomized
   cache-mutation oracle and the 3,335-victim recorded replay preserve decisions.
   The previous full recomputation dominated preliminary controller time.
10. All-to-all and all-gather connections are warmed before the 54GiB pinned
    host expert store is allocated. A padded R4 attempt made no forward progress
    for ten minutes in UVM memory operations while another CUDA job overlapped;
    sequential execution with explicit warmup completed and passed. This does
    not isolate the driver cause. No NCCL transport workaround is assumed.

## Correctness evidence

| Check | Result | Receipt |
|---|---|---|
| CPU regressions | 22 passed | `cpu_tests.log` |
| R1 native full model, 10% cache | 4 prompts, 768 MoE events, zero differences | `model_r1_bound_cache10/rank0.json` |
| R4 native full model, 10% cache | 4 ranks, 768 events, zero differences | `model_r4_bound_cache10/rank*.json` |
| R8 native full model, 10% cache | 8 ranks, 1,536 events, zero differences | `model_r8_bound_cache10/rank*.json` |
| R4 padded native batch, 10% cache | 32 questions × 4 generated tokens, 768 events, zero differences | `model_r4_padded_b8_warm/rank*.json` |
| R8 slots with substitution, Coverage, swap, resize | All outputs bitwise equal to independent FFN oracle | `slots_r8_incremental/rank*.json` |
| Recorded LRU policy replay | 96 events, 4,573 evictions checked | `replay_lru.json` |
| Recorded Gate policy replay | 96 events, 4,169 evictions checked | `replay_gate.json` |
| Recorded Coverage policy replay | 96 events, 3,335 evictions checked | `replay_coverage_incremental.json` |

Full-model parity checks selected expert IDs, routing weights, every MoE output
element/checksum and all generated token IDs. Four fixed prompts each produce
four tokens; this proves those cases, not arbitrary workload bitwise identity.
Every debug layer checks single-copy residency, physical tensor addresses and
physical fetch count versus logical misses. Slot fixtures also cover an empty
origin rank, an all-empty event, repeated eviction and alternate CUDA streams.

Replay takes the reference's recorded admissions to isolate substitution and
eviction. It verifies Hit/SubHit/Miss, merged gate mass, effective expert demand,
owners and each victim. It does not claim Python and the older simulator use
the same random generator or admission trajectory. Admission invariants cover
all seven policies; incremental swap decisions match a separate full-objective
recomputation over randomized cases.

## Physical transfer evidence

Nsight Systems 2024.6.2 measured the same real 9MiB-expert fixture after fixing
slot binding, using copy mode and direct-view mode. Both passed exact output
and physical slot audits. Each executes 155 experts, of which 48 require a
host fetch. The independent oracle's copies are excluded by runtime NVTX ranges.

| Mode | Actual expert H2D | Actual expert D2D | H2D/GEMM overlap |
|---|---:|---:|---:|
| Bound-slot copy baseline | 452,984,832 B | 1,462,763,520 B | 0.321 ms |
| Direct slot views | 452,984,832 B | 0 B | 0.420 ms |

H2D exactly equals logical misses × 9MiB. The old, incorrectly bound path also
showed host-to-device transfers for all three projections on every execution.
The new default eliminates that bypass and the full-expert D2D copy. Overlap
exists but is small in this fixture (4.94% of H2D time with views); this is not
a model throughput improvement claim. The controller pins active experts,
so these runs use direct fetches; the staging path is not forced by an unsafe
active-slot eviction.

Receipts: `profile_slots_bound_{copy,views}_summary.json`, corresponding
`.nsys-rep`/SQLite files, and `scripts/summarize_slot_trace.py`.
`fabric_warm_r{1,2,4,8}/rank*.json` separately records OS-local pinned H2D,
H2D+staging-D2D and NCCL all-to-all bandwidth with five samples per rank.
These are payload bandwidth measurements, excluding wire/protocol overhead.

## Model measurement matrix

Pending final aggregation in `model_matrix_v2`. The earlier `model_matrix`,
`benchmark_r4_probe` and `native_quality_screen` are explicitly superseded:
measurements predated the padding corrections and whole-generation throughput
clock. Their receipts remain on disk, excluded from final measurements. `scripts/run_server_matrix.py` defines sequential
R4/R8 A/B/C anchors at local batch 8/cache 30%, followed by local batches
4/8/16/32 × cache 10/20/30/40/50% with the configured Coverage + same/path
default. The protocol uses cold expert caches, fixed-step greedy generation,
explicit input/code fingerprints and separate instrumented ablation runs.
The short zero-shot numeric quality screen is paired with a newly generated
native control on the same questions. It is not the prior few-shot protocol
or a standard GSM8K test-set accuracy claim. No final speedup or quality-safety
claim should be inferred from the preliminary probe.
