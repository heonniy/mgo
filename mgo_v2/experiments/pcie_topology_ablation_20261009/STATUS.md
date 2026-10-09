# Execution status

The owner goal remains ACTIVE. Base: 7ddc55dff8d8a571a79a7fa415ad16fbc47e78dc. All new GPU work uses physical 0,1,4,5; group [0,1] on NUMA0 and [4,5] on NUMA1. Conda: /data2/esjung/envs/mgo-pcie. Model/data/results live under /data2/esjung. Owned burn is stopped under the exclusive experiment lease and restored while idle.

Completed and committed:

- G0 actual shared, fully pinned expert source: 54 GiB per NUMA node, shared by its two ranks; unique total 108 GiB, page placement and same-inode sharing verified.
- G1 native C++ quota/placement/admission/eviction controller parity; 4,112 quota cases and policy/cache differential replay.
- G2 all 41 H2D cells with 144 MiB sources and corrected physical-core affinity.
- Requested additional G2 all 41 cells with actual shared 54 GiB pinned sources: compact-address and full-pool-spread controls. Six-fetch 4:2 to 3:3 latency reduction: 23.51%, 24.01%, 23.82%, respectively. See microbench_pool_comparison/RESULTS.md. Commits 09c67d9, 5280f3b, 81562dc.
- G3 isolated physical phase ordering and full64 validation.
- Stage-I R-NEAR and G-NEAR: three unprofiled full64 repeats each, separate diagnostics, exact per-arm repeat token/cache parity. Median TPOT 1.943698 vs 1.930163 s, ~0.70% lower for G; no claim of a large robust generation gain. Commits 4882e08 and 5a88c09.
- Controlled Stage-II replay: all four G arms receive the identical per-event cache snapshot, miss keys and rank quotas over 3,072 events. This is conditional placement evidence, not serving TPOT. Commit 8c5dd21.
- Separate R/G CPU-call and CUDA-kernel breakdown; reference tokens/cache exact, all-rank phase-order verifier PASS. Raw Chrome traces compressed losslessly in stage1_breakdown. Commits 02239ea, 766c40e, 3439eaf, e7e12f4.
- Requested expert-compute follow-up: R-NEAR paired native/grouped on identical real packets, 48 events × 3 pairs × 4 ranks. Max-rank local completion median 12.1415 ms vs 0.41446 ms; ratio 29.29x applies ONLY to this compute packet set. All 192 weighted-output checks finite, maximum relative L2 0.4492%. Native output alone advances the model, so all reference tokens/cache remain exact. Actual grouped greedy serving and Ready-First performance are not yet measured. See grouped_same_packet/RESULTS.md and grouped_compute_probe.pdf. Commit d1a62fe.

Current owner priority and running work:

- Owner explicitly confirms grouped placement / naive placement means G-NEAR / R-NEAR. TPOT component diagnosis and fixes take priority over baseline execution. See TPOT_OPTIMIZATION_PLAN.md.
- G4_stage2_attempt1 was intentionally interrupted during warmup; no Stage-II primary completed. Private llama CPU build was also paused. Owner-interruption receipts are preserved; these are not unexplained runtime failures. Earlier worker exit/failure logs remain intact.
- G4_tpot_optimization_attempt1 is running on 0,1,4,5 with burn stopped under the experiment lease. Six arms: R/G native, R/G all-ready grouped decode, R/G grouped decode plus C++ Gate history/histogram and fused current-metadata GPU packing. Shared 54-GiB pinned source per NUMA, C30/P2 and globally serialized communications remain enforced. Each arm gets a disjoint full64 warmup, three counterordered unprofiled primaries (two more if any TPOT spread exceeds 5%), and a separate timers-only breakdown. C++ metadata CPU differential gate passed 176 updates with exact FP64 sums and FP32 scores; GPU wire, grouped actual-generation and timing results remain pending. Implementation commit 85fb226.

Remaining:

- First finish, validate and commit each current TPOT optimization arm and common R/G before/after breakdown; report actual placement benefit and any numerical trajectory changes.
- Then finish/validate/commit all Stage-II live arms and report communication/work imbalance, native controller and phase costs, numerical agreement and changed live cache trajectories separately from fixed-miss replay.
- Build and validate the frozen repaired MoE-Infinity, DeepSpeed ZeRO-Inference and balanced3 synchronous llama.cpp baselines on this host; run their frozen global64 primaries, preserve/repair failures, commit each result. Portable bounded baseline launcher and data2/NUMA paths prepared in 4f1946e. No baseline primary has run yet. Conda CUDA12.1 static runtime installed for llama build; no Torch/driver upgrade.
- Native-overlap supplemental Near comparison if feasible under the owner communication constraint; no Ready-First claim from serialized results.
- Complete final main_table CSV/LaTeX, RESULTS.md, required PDFs/phase/assignment artifacts and commit index. Do not mark the goal complete before required work is done.
