# DeepSeek C20–C50 Plateau: Next Bottleneck Diagnostics

**Status:** PLAN ONLY; no new GPU measurements, runtime modifications, or policy promotions.
**Parent evidence:** [RESULTS.md](RESULTS.md), [CACHE_ACTIVITY.md](CACHE_ACTIVITY.md), [TOKEN_STABILITY.md], [CACHE_PLATEAU_DIAG_RESULTS.md](CACHE_PLATEAU_DIAG_RESULTS.md), and [CACHE_PLATEAU_DIAG.json](CACHE_PLATEAU_DIAG.json).
**Reference commit:** `2e293717212a10fec8f7e8da81bc59da752caf5c`.
**Target:** DeepSeek-V2-Lite-Chat, ShareGPT 64-request batch, local B16, input 512, output 64, BF16 greedy, 26 routed MoE layers, four ranks on **physical GPUs 0/1/4/5 ONLY**.
**Goal:** Identify *exposed decode critical-path* costs, not merely the number of H2D bytes, before changing the quota/placement method.

## 1. Established observations (do not reinterpret as causal attribution)

| Metric | C20 | C50 |
|---|---:|---:|
| Primary TPOT median, 3 unfiltered repeats | 287.904 ms/token | 281.710 ms/token |
| Diagnostic decode-only H2D, 4 ranks | 1,397.5 GiB | 878.4 GiB |
| Maximum rank-mean explicit CUDA-stream H2D wait | 0.327 ms/token | 0.037 ms/token |
| Expert groups across all ranks / 63 decode intervals | 102,818 | 102,870 |
| Native ready-first execution waves across all ranks / 63 decode intervals | 23,908 | 13,423 |

- Decode H2D traffic fell **37.1%**, while primary TPOT improved **2.15%** (6.19 ms/token). This is not contradictory: current-layer demand copies run on a dedicated CUDA stream and ready experts execute without waiting for unrelated copies.
- The explicit `wait_for_slot` event is a narrowly measured main-stream dependency, **not** total DMA time, PCIe/HBM interference, CPU overhead, or a proof that all fetch-related effects are negligible.
- Almost the same number of executed expert groups but ~44% fewer execution waves at C50 is a specific **hypothesis to test**, not established causality; readiness, placement, and routing can all change.
- Instrumentation increased diagnostic TPOT by 3.2–3.8%; diagnostic phase spans include peer waiting/host submission and must **not** be added as if disjoint.
- C20 and C50 have the same input batch but only **25/64 complete output sequences agree**. Cache-dependent BF16 order can change subsequent token/routing trajectories. The current difference is a valid per-capacity greedy-run result, *not* a controlled same-route attribution.
- `route_to_dispatch_host` ranged **58.8–72.8** (C20) vs **58.5–86.4** ms/token (C50), expert spans **85.3–93.7** vs **72.6–90.2**, and return/combine **76.8–82.2** vs **79.1–96.6**. These per-rank spans are not isolated component service times.

## 2. Questions and competing hypotheses

1. **H2D actually hidden?** Is the exposed DMA dependency small under a matched route and when indirect PCIe/HBM contention is controlled? Does no-copy execution materially outperform demand-copy execution?
2. **Ready-first wave overhead?** Does lower cache residency fragment native execution into more `execute_wave` crossings, `ready_many` polls, and tiny individual GEMMs, even when explicit waits are tiny?
3. **Host orchestration overhead?** How much time is spent per layer in all-gather/D2H metadata, CPU placement, packet layout, slot binding, H2D enqueue, and GPU packet construction?
4. **Critical rank and collective arrivals?** Are return/combine spans predominantly waiting for the last rank, repeated `index_add_` work, or actual NCCL transport service?
5. **Capacity-induced workload changes?** Are apparent changes driven by different expert owners, group counts, numerics, and future routes rather than the absence of H2D?

No single hypothesis above has yet been established as the remaining TPOT bottleneck.

## 3. Measurement contract / correctness

- **Baseline anchor:** exact commit/configuration/manifests from the original C20/C50 primary runs; preserve prior results and paths. Never overwrite the primary 3-repeat rows.
- **Paired workload:** run a clean same-input A/B pair first; then an independent *fixed-continuation* pair in which each request's next token IDs are supplied from a single reference continuation rather than its live argmax. Check all token inputs across steps match.
- **Routing remains a possible confound:** fixed token IDs alone do **not** guarantee identical router IDs/probabilities when BF16 execution order differs. Capture per-step/per-layer/per-rank selected-expert and demand hashes. If they diverge, add a separate frozen-router replay (selected IDs, routing weights, full router probabilities for gate history) and label its result **controlled route replay**, not normal generation or an accuracy comparison.
- Record global active-expert sets, per-rank owners, group counts, per-rank expert token rows, dispatch/return packet counts and bytes, mandatory fetches, cache evictions, and H2D bytes. Verify cache ownership, transfer slot readiness, active-expert protection, collective transpose, token/route hashes, and no NaN/Inf.
- Report **per-layer/per-step** max-rank completion and arrival skew before aggregation; the slowest rank can change by layer. Keep both rank-level and global results; never sum independently timed overlapping CUDA events or interpret an all-to-all completion span as pure wire time.
- Separate **clean primary timing** (>=3 repetitions per arm, alternating order, no detailed per-layer event capture) from **sampled diagnostic timing** (short prefix, minimal instrumentation, no added per-layer GPU synchronizations). Report profiling overhead against clean runs.
- Record host contention/load, NUMA affinity, physical GPU IDs, topology, memory usage, source revisions, command line, trace hashes, CUDA/NCCL environment, and CI/test receipt for each run.
- Raw token IDs, full traces, profilers, and large arrays remain in guarded external results, **not Git**. Commit only manifest/hash references, validated aggregate CSV/JSON/MD, and reproduction commands.

## 4. Ordered diagnostic stages (no automatic execution)

### D0 — Offline replay and trace equivalence (NO GPU)

- Audit existing C20/C50 reports and per-rank diagnostics; recompute the numbers in Section 1.
- Validate C20/C50 per-rank expert-group and wave counts (and normalize by 26 layers × 63 decode intervals).
- Design a frozen continuation and frozen-router manifest; specify a consistency check that fails closed on route-hash mismatch.
- Inventory the DeepSeek hot paths: `examples/headline_ours_deepseek_worker.py`, `mgo_v2/live_metadata.py`, `scripts/env_offload_policy.py`, `scripts/env_offload_rank_layout.py`, `mgo_v2/deepseek_native_expert.py`, `mgo_v2/csrc/native_expert_deepseek.cpp`, `mgo_v2/fused_transport.py`, and `mgo_v2/pinned_h2d.py`.
- Deliver `D0_AUDIT.md` and exact manifest specifications before requesting GPU time.

### D1 — Matched-route C20 vs C50 (highest priority; GPU approval required)

- Repeat both capacities on the *same* decode-token continuation; additionally capture/evaluate a fixed-router version if numeric paths diverge.
- Distinguish globally active experts (fixed under route replay) from rank-assigned groups (can differ with cache/placement), ready-first waves, and actual H2D bytes.
- Report paired TPOT, layer-level max-rank duration, completion/arrival skew, H2D dependency, wave count, and controller time. Separate each policy's deterministic repeat from the cross-policy controlled comparison.
- **Pass:** exact input and selected-route hashes, demand/ownership invariants and an unchanged model/output shape; never silently accept routing divergence as a matched-route result.

### D2 — Full-resident/no-demand-H2D reference (GPU approval and memory preflight required)

- Create a **diagnostic-only**, fully resident reference using the same model, token inputs, attention/KV, EP dispatch/return path, numerical precision and expert executor. Explicitly preload all 1,664 routed experts with a unique, deterministic owner map before measured decode, and verify zero demand H2D.
- For a four-rank partition, full residency needs **416 MAIN experts per rank**, *in addition to any reserved scratch/staging slots*. Do not incorrectly call a 416-total-slot allocation with two reserved scratch slots `C100`.
- A static full-resident owner map **changes placement and EP workload**, so its TPOT is only a **reference**, not a causal upper bound on the speedup of removing DMA from the C20/C50 schedules.
- If memory and implementation permit, add a separate **schedule-preserving no-copy oracle**: replay exactly the baseline expert owner/dispatch schedule using preloaded physical weights while emulating the original *logical* cache state and admission. Label any extra physical replicas/oracle HBM as diagnostic resources outside the advertised cache budget. This is the strongest copy-removal counterfactual, but it is optional and must not skip correctness checks.
- Optionally compare fully resident execution with vs without **synthetic demand-copy traffic to an isolated scratch buffer**, matching per-layer bytes/timing where feasible. This can test indirect PCIe/HBM interference while preserving computed expert weights. It is a separate ablation, not the normal runtime.
- **Pass:** same route hashes (or explicitly report that they differ), exact owner/packet replay for the schedule-preserving arm, no unauthorized memory pressure, no in-measurement weight fetch in the no-copy arm.

### D3 — Exposed critical-path profiler (short sampled decode, GPU approval required)

Per **(step, layer, rank)** capture, at minimum:

| Part | What to separate |
|---|---|
| Metadata | router IDs/probabilities pack, NCCL metadata all-gather, D2H handoff / synchronization, gate history |
| Controller | `Policy.apply` and NEAR decision, victim/cache updates, native slot binding |
| Layout | rank-partial/index plan and packet packing, scatter/gather kernels |
| H2D | enqueue-to-submitted delay, DMA start/end events, queue depth, overwrite dependencies, explicit main-stream wait, aggregate traffic |
| Executor | `ready_many` query/poll CPU cost, number of ready waves, `execute_wave` host boundary, expert groups, token rows, per-expert GEMMs, GPU kernels |
| EP | dispatch A2A start/completion, return A2A start/completion, rank arrival skew and minimum service floor |
| Combine | rank-local partial `index_add_` launches and source-side accumulation/reduction |

- Use Nsight Systems/NVTX or low-overhead CUDA events without introducing rank barriers in the normal path. After each layer, reconstruct *which rank* sets the expert-to-return critical path; aggregate layer maxima, not only rank-total maxima.
- Preserve **CPU wall spans vs CUDA-stream spans vs peer-wait-inclusive collective spans** as different columns. Report profiler overhead and keep raw timelines off Git.
- Distinguish pure GEMM/kernel occupancy from Python/C++ orchestration. The present native C++ loop batches the **host call**, but still runs separate gate/up/down GEMMs for each expert; do **not** describe it as grouped GEMM.

### D4 — Mechanism isolation (only after D1–D3)

Run one controlled change at a time on the same frozen route:

1. **Ready-first fragmentation:** quantify the relation of waves, `ready_many` CPU polls and native host crossings to latency. Try a scheduler-only grouping change while preserving dependency safety; treat any BF16-order changes as a separate correctness concern.
2. **Expert kernel overhead:** if GEMMs dominate, prototype a true grouped/persistent MoE executor or low-launch-overhead GPU kernel. Compare both standalone GEMM microbench and full clean TPOT; do not confuse fewer Python calls with grouped GEMM.
3. **Rank-partial combine:** profile repeated `partial.index_add_` and final output reduction; prototype fused/segmented reduction with explicit BF16 numerical checks.
4. **Metadata/placement:** split all-gather, D2H, controller, and layout into distinct host and stream spans; optimize only the dominant component. A fixed-route no-op placement **diagnostic** may estimate controller headroom but is not an equivalent inference system.
5. **Return arrival skew:** distinguish synchronization/late peer arrivals from actual NCCL service. A barrier ablation can characterize waiting but adds its own overhead and is not by itself an optimization.
6. **H2D interference:** investigate if the no-copy reference or synthetic DMA shows a robust difference despite near-zero explicit wait. Only then prioritize PCIe bandwidth/topology-aware quotas.

Each ablation needs matched-route validation, at least three uninstrumented timings, full min/median/max, and checks that it does not silently change EP packet format, effective routes, precision, or global cache capacity.

## 5. Decision gates for the next research direction

- **H2D path still matters:** matched-route no-copy execution improves clean TPOT materially **after controlling for owner placement**, or DMA-injection measurably slows a resident schedule. Investigate indirect contention and topology-aware transfer scheduling; direct wait alone is insufficient evidence.
- **Execution-path bottleneck:** profiling shows many tiny GEMMs/launches or persistent waves causing the layer max rank to stall, and executor ablation reduces clean TPOT. Prioritize grouped expert execution / wave scheduling.
- **Control-path bottleneck:** metadata D2H, `Policy.apply`, layout construction or enqueuing occupies the observed critical path and a targeted change lowers clean TPOT. Optimize control/layout rather than changing the fetch objective.
- **Collective/straggler bottleneck:** return completion follows late rank arrivals; rank-group imbalance or partial-reduction launches matter more than peer byte count. Refine a calibrated critical-rank placement proxy or fuse reductions.
- **No decisive result:** do **not** promote policy/runtime changes or claim the residual TPOT gap is caused by a single stage. Recheck trace equivalence, overhead, and repeatability.

For a promotion request, require a repeatable **clean** TPOT improvement larger than observed run-to-run noise, preserved correctness and capacity invariants, and no regression in E2E/TTFT or the main baseline cells. Never select a winner using instrumented TPOT alone.

## 6. Safety, reproducibility, and stop boundary

- **Allowed physical GPUs: 0, 1, 4, 5 only. GPUs 2, 3, 6, 7 are reserved for another user: never probe, allocate, benchmark, or synchronize them.**
- Sequential guarded jobs only; preserve fixed CPU affinity/NUMA settings, external raw-output locations, prior warmup/target separation and source hashes. Before any full-resident oracle, preflight per-rank expert HBM plus KV/dense/scratch headroom, pinned-host memory, and allocation fallback.
- Do **not** launch GPU measurements, alter runtime behavior, change policy defaults, perform automatic C100 sweeps, or claim unmeasured speedups as part of this PLAN commit.
- Deliver an implementation plan and request owner approval for the GPU diagnostic set before running it. The current verified conclusion is limited to: **directly exposed H2D waits are small under the measured C20/C50 conditions; the rest of the critical path is not yet isolated**.

## 7. Expected artifacts when approved

`D0_AUDIT.md` (offline); `MATCHED_ROUTE_VALIDATION.json`, `NO_COPY_ORACLE_RESULTS.md`, `LAYER_CRITICAL_PATH.csv`, `EXECUTOR_WAVES.csv`, `PHASE_BREAKDOWN.md`, and `FOLLOWUP_DECISION.md` (subsequent measurements). Include commands, file hashes, primary-vs-diagnostic overhead, and numerical/correctness validation; do not commit raw token IDs or large profiler traces.
