# FERRY Appendix Validation — experiment plan (2026-10-11)

**Status: PLAN ONLY — not executed.** This document defines the missing appendix evidence and reuses existing receipts when possible. No performance figures below are newly measured. The requested deliverable is a reproducible plan and an appendix LaTeX draft, not a change to the selected runtime policy.

**Repository / starting ref:** `heonniy/mgo`, `paper_result` (inspected at `a0d77a97ea5feb0406e62702962c377f4c2ed365`). **Suggested plan path:** `mgo_v2/experiments/ferry_appendix_validation_20261011/PLAN.md`. Work on an isolated experiment branch for implementation and run only after explicit authorization; do not silently change `main_OURS`, the main table, or default cache/admission policies.

## 0. Research questions and required appendix artifacts

| ID | Question | Additional experiment? | Expected artifact | Priority |
|---|---|---|---|---|
| E1 | Why routing probability instead of selection count? | Yes; matched trace & score comparison | Future activation calibration, Recall@K, cache hit/miss, optional TPOT | P0 |
| E2 | Why use `W=128`? | Yes; token-window sensitivity and order check | Window sweep with rank-order robustness; no step/window confusion | P0 |
| E3 | Why `delta=max(0.02 P*,1)` for Near? | Yes; matched placement sensitivity | Compute skew, locality, collective latency, TPOT vs tolerance | P0 |
| E4 | Does topology-aware bandwidth profiling predict transfer completion? | Reuse existing measurements, targeted check if needed | Domain/microbenchmark table, predicted-vs-measured fetch time, quota example | P1 |
| E5 | What is metadata exchange overhead? | Optional focused diagnostic | Payload bytes and per-phase metadata time, parity/consistency | P1 |
| D1 | What metadata does each stage need? | No GPU run | Appendix table: local/exchanged/derived/offline state, owner, update scope | Documentation |
| D2 | How are transfers profiled and quotas selected? | No new run for pseudocode | Offline profiling algorithm + worked example | Documentation |
| D3 | How were experiments and baselines configured? | Audit receipts, not an E2E rerun | Hardware/model/memory/pinned/baseline config tables | Documentation |

## 1. Existing evidence: use before measuring again

- `mgo_v2/experiments/main_eviction_history_20261007/{PLAN.md,RESULTS.md}`: CPU-only, frozen-trace Qwen3 C30 BR comparison, gate-score W128 vs LFU-reset/cumulative vs LRU-reset/cumulative. *Important*: LFU counts one use per **distinct expert/layer event**, not a token-level Top-K selection frequency. Decode-only gate-score hit rates are 42.41%/35.34%/29.95% for local B8/B16/B64 in that **specific** setting. These are distinct-expert cache-hit rates, not per-token probability prediction or E2E improvements.
- `mgo_v2/experiments/deepseek_eviction_hit_20261009/{README.md,RESULTS.json}`: DeepSeek frozen-trace C30/C60 CPU replay. It uses different input/decode settings and a source captured under C20; do **not** directly average/model-compare absolute hit percentages with Qwen.
- `mgo_v2/experiments/pcie_rank_concurrency_20261009/{RESULTS.md,RESULTS.json,RANK_COMBINATIONS.csv}`: measured pinned H2D concurrency and rank subset effects for Qwen and DeepSeek expert payload sizes; indicates bandwidth heterogeneity. Unit in this report is **GiB/s**, whereas main-paper Table 1 labels **GB/s**: normalize before reusing numbers.
- `mgo_v2/experiments/pcie_quota_r8_b16_20261010/{REPORT.md,LOOKUP.json}`: 255 nonempty 8-GPU subsets, lookup calibration, 12-miss and 68-miss **predicted fetch-time** examples. The calibrated lookup **did not establish an end-to-end TPOT gain** over simpler quotas; do not characterize predicted fetch gains as model gains.
- `mgo_v2/experiments/pcie_quota_grouped_r8_b16_20261010/GROUPED_RESULTS.md`: follow-up grouped regime also had overlapping TPOT ranges; do not select a best seed/post-hoc winner.
- `mgo_v2/experiments/native_ca_near_20261008/RESULTS.md`: Near vs communication-oriented CA in serial executor; placement changes expert/collective critical paths. This is **not** a tolerance sweep.
- `mgo_v2/experiments/main_table_2x2_20261008/{PLAN.md,MEMORY_AUDIT.md}`: exact models, 30% expert budget, BF16, pinned host memory audit, baseline settings, dataset manifest protocol, and 3-target-repeat plan. Main-paper Table 2 currently says **mean ± sample standard deviation**, while the repo PLAN specifies **median and full ranges**. Reconcile the publication's statistics explicitly against raw receipts before finalizing Appendix.

## 2. Cross-experiment requirements (apply to E1–E5)

**Reference configuration:** physical GPU IDs `0,1,4,5` only for R4; Qwen3-30B-A3B-Instruct-2507 and DeepSeek-V2-Lite-Chat; ShareGPT primary, LMSYS-Chat-1M if feasible; 30% expert-cache budget (C30); BF16; greedy generation; input 512, output 64, local batches 16 and 64. Prefetch OFF, substitution/replication OFF for the isolated eviction and placement question unless the selected paper setting differs. Freeze and record cache policy/admission and execution mode; do not silently combine serial and grouped executor results. Use each model's own checkpoint and chat-template token manifests.

- Pin code ref/checkpoint hash, tokenizer/chat-template hash, CUDA/PyTorch/transformers/native-extension versions, physical GPU IDs, CPU/NUMA affinity, memory budgeting, pinned host allocation strategy, and relevant environment flags.
- Use a **single matched source trace** when comparing eviction or placement policies. Record layer/event IDs, origin ranks, Top-K expert IDs, full pre-TopK router probabilities, and any cache hits; verify expert ID range and global demand conservation. Keep private prompts, token manifests and large routing traces outside Git; push only hashes and aggregate anonymized measurements.
- Separate CPU deterministic counterfactual (same exact trace, no floating-point model divergence) from real-GPU timing; do not label replay hit-rate changes as TPOT improvements. Validate cold-cache initial state and whether prefill populates cache and eviction history.
- For real-GPU timing, use disjoint warmup and target batches, clear the dynamic cache between target repeats, and preserve **all** repeats and ranges. Main-table protocol calls for three clean target repeats; use at least three here if comparing TPOT, rotate policy order, flag large noise and do not drop outliers without a predefined, identical rule. Do not use an exploratory seed as confirmatory evidence.
- Report both **distinct expert-request hit rate** and **token-weighted hit rate** where relevant, with denominators and prefill/decode explicitly stated. Report H2D bytes/fetches, critical rank expert rows, collective completion spans and whole-token TPOT. Do not sum overlapping instrumented intervals.
- Respect current repository safety instructions in `mgo_v2/AGENTS.md`. The previous owner-managed background burn workload has been disabled; do **not** restore it. Do not touch GPUs `2,3,6,7` without a fresh explicit authorization, even though earlier R8 results exist. No GPU jobs are authorized by this documentation commit.

## 3. E1 — Routing probability vs. token-level selection count

**Question.** Does pre-TopK probability retain useful information about experts *nearly selected* by the router, beyond whether an expert appeared in the actual Top-K?

**Critical definition.** For each layer and the same past-`W` token set `H`:

- `s_e = sum_{i in H} p_{i,e} / |H|`, using **full normalized router probabilities before Top-K masking**.
- `c_e = sum_{i in H} 1[e in TopK_i] / |H|` (**token-level Top-K count**, not distinct-event LFU).
- Optional supplementary comparators: event-level LFU (`1` per `(step, layer, expert)`), LRU, random. Keep tie-break rules identical (LRU then expert ID), and pin all currently active experts.

**Measurements.** Use horizon `h ∈ {1,4,8,16,32}` following decode steps for *future* required experts, without looking ahead when calculating the score. Report future-occurrence probability by score quantile, Spearman rank correlation, Recall@K or NDCG@K of future expert demand, number of distinct scores/ties, cache-hit and miss counts in deterministic identical-trace replay, and optionally repeated full-GPU TPOT for `W=128` probability/count/LRU. Split analysis by model, layer, local batch and score horizon.

**Success/failure interpretation.** Probability is justified only if it demonstrably ranks **future** demand or reduces misses relative to token-level selection count at matched W, not merely because its values are continuous. If routing probabilities are already Top-K-masked, the “nearly selected” argument must be removed/rewritten. Existing LFU study is supportive but **not a substitute** for this comparison.

**Implementation candidates:** `mgo_v2/mgo_v2/eviction.py` (`GateHistory`, `GateScoreEviction`), `mgo_v2/mgo_v2/live_metadata.py`, trace replay scripts from `main_eviction_history_20261007`. Suggested new scripts: `capture_evict_trace.py`, `replay_score_baselines.py`, `evaluate_future_demand.py` (not currently created).

**Outputs:** `ROUTER_SCORE_RESULTS.csv`, `FUTURE_DEMAND_CALIBRATION.csv`, `CACHE_REPLAY_RESULTS.csv`, `FIG_PROBABILITY_VS_COUNT.pdf`, `RESULTS.md` + provenance and invariant checks.

## 4. E2 — History window W=128: token semantics and robustness

**Important correction:** Current implementation (`GateHistory.update`, `live_metadata.py`) tracks the **most recent W token-probability vectors for each layer**, *not W decoding steps*. With R4 × local B64, a single layer event contains 256 tokens; `W=128` can cover only part of that event. With R4 × local B16, W128 spans approximately two global layer events. Do not call it “128 decode steps” in the manuscript.

**Sweep:** `W ∈ {32,64,128,256,512}` tokens, optionally `W=1024` for a longer-history anchor. Freeze trace and cache/admission budget, use the same history initialization and tie-breaking. For each W, report online score variability, future-demand prediction metric from E1, cold-start behavior, distinct/token-weighted cache hits, H2D fetch count, and optional actual TPOT (repeat only competitive W settings).

**Required rank-order bias check.** The routing metadata is concatenated in deterministic rank order. When `W < global_batch`, the “latest W tokens” may be dominated by the last ranked GPU(s) **within the same decode step**, despite simultaneity. Hold each event's tokens fixed and compare the score/cache outcomes under several permutations of rank concatenation and under a rank-balanced W-token sample. Document absolute differences in evictions, hits and TPOT (timing optional). If order changes results materially, either change the policy definition to an event-aware/rank-balanced window or state the limitation; do not claim a universally optimal W=128.

**Outputs:** `WINDOW_SENSITIVITY.csv`, `RANK_ORDER_ROBUSTNESS.csv`, `FIG_WINDOW_SWEEP.pdf`, `RESULTS.md` (with model/batch conditioning).

## 5. E3 — Near tolerance sensitivity

**Definition from paper §4.3:** `P_r(e)=max(max_j N_j, N_r + d_e)`, `P*=min_{r in Q}P_r(e)`, `delta=max(alpha*P*,1)`, `R_e={r in Q: P_r(e) <= P*+delta}`; chosen rank maximizes `d_{r,e}`, with paper's tie breaks. Baseline `alpha=0.02`.

**Sweep:** `alpha ∈ {0,0.01,0.02,0.05,0.10}` with the integer floor fixed at 1. Include strict `delta=0` as a *separate* diagnostic because `alpha=0` still gives `delta=1`; otherwise one might falsely call it strict minimax. All policies must receive exactly the same quota vector and frozen global miss set for each layer event. Keep memory/batch/executor fixed.

**Metrics:** per-layer peak projected vs actual expert-token rows, expert-group count/slowest-rank compute interval (grouped mode may not scale with rows), rank compute imbalance, remote token fraction, NCCL submitted payload bytes, return/dispatch completion including synchronization, controller overhead, same-trace cache/fetch counts, and repeated whole-token TPOT.

**Stratify:** Qwen & DeepSeek, at least local B16/B64, `C30`; serial executor and grouped executor are distinct regimes, not pooled. Existing R8 grouped data suggest Near need not beat random; present negative results if reproduced. Avoid blanket “Near is optimal” claims.

**Outputs:** `NEAR_TOLERANCE.csv`, `NEAR_TRADEOFF.pdf`, `RESULTS.md`, parameter-definition/unit tests.

## 6. E4 — Profiling accuracy and worked quota example

**Known implementation:** paper Eq. (6) models `T_fetch(q)=S max_{g in G} sum_{r in g}q_r/B_g`; Eq. (8) chooses feasible integer quotas minimizing this surrogate. `B_g` may represent nested shared resources, not only individual GPU links. Physical `S` and measured `B_g` must have matched byte/time units (GB/s vs GiB/s).

**Documentation-only first:** derive a reproducible profiling procedure: (i) discover candidate topology groups; (ii) allocate pinned expert-sized sources; (iii) calibrate individual-rank transfers; (iv) calibrate simultaneous transfers for candidate shared groups; (v) record effective `B_g`/lookup and grouping rule; (vi) greedily allocate `q` by predicted `T_fetch(q+1_r)`, respecting free/evictable slot bounds `u_r`.

**Reusable physical example:** the R8 12-miss report uses standard balanced `[2,2,2,2,1,1,1,1]` and measured-bandwidth-aware `[1,1,1,1,2,2,2,2]`; predicted latest-rank fetch completion is **0.536 ms vs 0.429 ms**. Label both values as **profiling-model predictions**, not actual TPOT. On the same report, the full-model calibration lookup showed **no stable TPOT improvement** over a simpler fast-rank heuristic. This example is supplementary R8 scaling evidence, not the main R4 benchmark.

**New measurement only if required:** compare predicted and *measured* burst completion for a held-out set of R4 quota vectors, with a fixed expert payload, fixed active sets and 3+ repetitions. Include synthetic short/long burst lengths and ensure each bandwidth group is a measured transfer resource rather than an assumed NUMA assertion. Report model MAE/MAPE and calibration condition, and show whether a predicted faster quota is measured faster. A simple model can be imperfect due to transfer startup, overlap, host contention and virtualization; do not assert Eq. (6) is exact.

**Outputs:** `BANDWIDTH_DOMAINS.csv`, `QUOTA_EXAMPLE.csv`, `FETCH_PREDICTION_VALIDATION.csv`, `PROFILE_ALGORITHM.md`, `RESULTS.md`.

## 7. E5 — Metadata composition, correctness and overhead

**Logical metadata table (to be included in Appendix):**

1. **Local routing input:** current `(t, layer, rank)` event ID, per-source token count/identity, per-token selected Top-K expert IDs, full router probabilities for gate history (where needed).
2. **Aggregated demand:** `d_{r,e}` (source-rank × expert token counts), `d_e=sum_r d_{r,e}`, required expert set `E={e:d_e>0}`.
3. **Controller state:** resident owner/cache membership `C_r`, current active-expert protection, rank's free/evictable slots `u_r`, resident-expert compute load `N_r`, per-layer gate history `H^(layer)`.
4. **Offline system information:** bandwidth domain membership `G` and measured/lookup bandwidths `B_g`, expert weight size `S`.
5. **Derived decisions:** global misses `F = E \ union_r C_r`, quotas `q_r`, expert-to-rank mapping `F_r`, cache evictions and updates.

**Avoid mislabeling exchanged bytes.** `GlobalExpertController` is designed to be deterministic and replicated on all ranks; full cache state need not be sent in a new all-gather on every layer. The compact path `compact_metadata.py` packs `24 + B_local*k + 4E` bytes per rank (`uint8` Top-K IDs, float32 per-expert gate summary) and a 24-byte event header; the live path `live_metadata.py` packs `24+B_local*k+4E*min(B_local,128)` bytes, with a tail of full probability rows for exact rolling gate history. These are **two different paths** and should not be conflated or advertised as one universal payload. The exact packet path enabled for main-paper cells must be checked against run receipts.

**Optional diagnostic run:** use existing `LiveMetadata.collect_profiled` intervals for GPU pack, all-gather, GPU-to-host copy, CPU parse, gate-history update, and demand histogram, comparing actual packet bytes and event counts against formula. Report one layer-event median and p95, all ranks, and fraction of decode TPOT; instrumentation spans include synchronization and may overlap other work. Verify all rank event headers, histogram conservation `sum_{r,e}d_{r,e}=total_tokens*k`, gate-vector normalization/finite status, identical replicated residency decisions and no hidden future routing information.

**Outputs:** `METADATA_FIELDS.md`, `METADATA_BYTES.csv`, `METADATA_TIME.csv`, `PARITY_CHECKS.json`, `RESULTS.md`.

## 8. D3 — Appendix reproduction audit (no new GPU benchmark)

1. **Hardware:** KVM-virtualized server; Intel Xeon Platinum 8480+; H100 80 GB ×4, NVSwitch inter-GPU; PCIe Gen5 ×16 for host links. Record physical CPU socket/node layout if reliably available; do not infer connectivity from NVSwitch alone. PCIe line rate is theoretical, not observed H2D service.
2. **Models:** Qwen3-30B-A3B-Instruct-2507 (48 layers, 128 routed experts, Top-8); DeepSeek-V2-Lite-Chat (27 layers, 64 routed experts, Top-6, MLA and shared experts). Record exact model revision, precision, expert size bytes, dense/shared expert placement, cache slots, and model-specific attention kernels. Use model-config evidence rather than inferring all layers are MoE layers.
3. **Memory:** 30% GPU-resident expert-weight budget, dense attention/shared expert weights on GPU, KV cache separate. FERRY has **rank-private full-pinned host expert sources**: Qwen `54.00 GiB/rank`, DeepSeek `26.81 GiB/rank` in the R4 main-table memory audit. Baseline pinned bytes differ: DeepSpeed `14.22 / 7.31 GiB/rank` (Qwen / DeepSeek); MoE-Infinity and llama.cpp are **not measured** in that audit. Do not claim identical pinned allocations or confuse pinned host memory with HBM expert-cache occupancy. Record staging reserves and peak HBM separately.
4. **Workloads:** ShareGPT, LMSYS-Chat-1M, exactly 512/1024 input tokens, local B16/64 on each GPU (global B64/256), greedy ignore-EOS generation of 64 output tokens. Token-ID manifests are model-specific but shared across methods. Disjoint warmup and target, empty dynamic cache after warmup, results labelled TTFT/TPOT (attention included)/end-to-end TPS.
5. **Baselines:** DeepSpeed ZeRO-Inference with ZeRO-3 CPU parameter offloading; repaired MoE-Infinity with static expert ownership and independent rank caches; synchronous llama.cpp layer-wise CPU/GPU split (Qwen: balanced 3/3/3/3 expert-layer ownership, 32 threads, CUDA graphs OFF; DeepSeek: per-GPU one routed-expert layer with unused expert-budget capacity); FERRY-Random needs a precise definition of whether fetch quota and cache eviction are shared with FERRY. Use implementation-verified settings and recorded versions, not marketing names.
6. **Statistics reconciliation:** main-paper Table 2 caption says `mean ± sample standard deviation`; `main_table_2x2_20261008/PLAN.md` prescribes a 3-run `median and full ranges`. Retrieve original run-level receipts/CSV and resolve this mismatch before writing a definitive reproducibility claim in the paper. State whether outliers were filtered and by what preregistered rule; never recompute from rounded table cells.

**Outputs:** `APPENDIX_REPRODUCIBILITY_AUDIT.md`, `BASELINE_CONFIGS.md` (exact command+commit hashes as available), `MAIN_TABLE_STATS_AUDIT.md`.

## 9. Completion checklist and review gate

- [ ] E1: pre-TopK `p` vs **token-count** `c` on identical history and future route trace, with tie and future-leakage checks.
- [ ] E2: W token sweep and rank-concatenation order invariance check; document if W128 is not optimal in every regime.
- [ ] E3: Near alpha sweep; strict delta=0 distinguished from alpha=0 with floor1; analyze grouped vs serial separately.
- [ ] E4: reuse microbench receipts and add R4 held-out quota prediction if the appendix needs an accuracy claim; mark R8 numbers as model-predicted.
- [ ] E5: document exchanged vs local metadata, validate packet bytes and parity; optional isolated timing.
- [ ] D3: versions/pinned memory/model/baseline/config statistics audit; reconcile mean ± SD vs median range.
- [ ] Every run: provenance, 3+ target repeats for timing claims, all values/ranges, hashes/units, no raw private manifests in Git.
- [ ] Only after checks: insert empirical plot/table in appendix LaTeX. No unmeasured accuracy, correlation, or speedup claims.