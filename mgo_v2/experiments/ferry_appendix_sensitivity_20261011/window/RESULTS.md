# FERRY appendix: Gate-history window sensitivity (why W=128?)

Status: PASS. Three new GPU trace captures plus CPU-only replays. Nothing is committed.
This is a replay of cache metrics, not a timing result. A higher hit rate here does not
mean a lower TPOT, and no TPOT was measured.

## Question
The FERRY Gate-score MAIN eviction ranks residents by the mean of the most recent
**W token** router-probability rows of their layer. The rows are concatenated across
ranks in rank order 0..3, and W counts tokens, not decode steps. The setting under
test is W=128. Is 128 optimal, on a plateau, or arbitrary?

## Method
- **Workload.** This is the same cell as `main_eviction_history_20261007`: Qwen3-30B-A3B-Instruct-2507,
  R4 on physical GPUs 0,1,4,5, C30 (MAIN 461/461/461/460), input 256, a cold cache before
  prefill, 1 prefill forward and 256 decode forwards, BR admission with seed 42, prefetch OFF,
  gate-score eviction, and the same `WORKLOADS.json`. Local B8/B16/B64 correspond to global
  batches of 32/64/256 tokens per decode layer-event.
- **New capture.** This is opt-in only and needs `MGO_FERRY_WINDOW_PROBS=1`.
  `scripts/ferry_window_probs.py` is hooked from `RoutingCapture.install/finish` in
  `scripts/main_eviction_trace.py`. Without the env var the path is unchanged. Each rank
  wraps `rt.plan_event` and stores its own local float32 router-probability rows with no extra
  collectives:
  - every decode layer-event, holding exactly the rows that `LiveMetadata` sends;
  - per prefill layer, the last 1024 local rows, plus the float64 sum and count of all local
    prefill rows.

  Raw files total 2.4 GB and are kept outside git.
- **Gate traces.** For each W, the gate trace is rebuilt with the production
  `mgo_v2.eviction.GateHistory(window=W)`, using the same calls as the live runtime:
  - prefill uses `update(last W global rows)`;
  - decode uses `update_compiled(concat rows)`;
  - the score is `float32(sum/len)`.

  Anchors:
  - **all-history:** cumulative mean over all prefill and decode tokens;
  - **current-event:** mean of only this event's rows. At B8 this equals W=32 by construction.
- **Replay.** The replay copies `main_eviction_history_replay.replay` (mode 0) with the same
  BR quota/shuffle, seed 42, active pinning, and the score→last-use→key tie-break, adding only
  a token-weighted hit counter. It is asserted identical to the reference replay. Capacities
  are C30 [461,461,461,460], C40 [615,615,614,614], and C60 [922,922,921,921]. C40 and C60 are
  CPU counterfactuals on the same trace.
- **Metrics.** All counts are decode-only (steps 1..256):
  - distinct-expert hit rate (one request per step, layer and expert);
  - token-weighted hit rate (top-8 assignments served by a resident expert);
  - misses and evictions.
- **Prediction.** For each decode event (step s, layer l), the Spearman rank correlation between
  the score vector (which includes the current event, so there is no look-ahead) and the demand
  of layer l summed over steps s+1..s+h, for h = 1, 4 and 16, averaged over events.
- **Rank-order bias.** The W sweep was repeated with:
  - rank concatenation orders reversed (3210);
  - two seeded random orders, (0321) and (2103), drawn with `default_rng(20261011)`;
  - a rank-balanced window holding the latest W/4 tokens of each rank.

  Order 2103 keeps rank 3 last. So whenever W ≤ B_local it is identical to the identity order,
  which serves as a built-in sanity check.
- **Noise reference.** `SEED_NOISE.csv` replays W ∈ {32,128,1024} with ten BR shuffle seeds.
  The seed is a policy-neutral counterfactual perturbation.

## Validity checks (VALIDITY.json); all pass for B8, B16 and B64
1. Rebuilding W=128 from the raw rows reproduces the stored live gates **bit-exactly**
   (max |Δ| = 0, 0/12336 events differ).
2. Replay with the rebuilt W=128 gates at source capacity equals the live
   `reference_metrics` and `final_slots` exactly. The stored-gate reference-replay parity also
   passes, and the token-weighted replay copy equals the reference replay.
3. The top-8 of the captured decode rows reproduces `counts` in 12288/12288 events.
4. **Routing identity with the 2026-10-07 capture.** `counts`, `gates`, `reference_metrics` and
   `final_slots` are identical, and the `routing_trace.npz` file SHA256 is byte-identical to the
   old capture:
   - B8: 2bc2d9ee…
   - B16: 4f5dfa4b…
   - B64: 30f110b5…

   The new code base is a0d77a9; the old one was 7e5b410.
5. The W=128 identity row reproduces the published history table: 42.41 / 35.34 / 29.95 %.

## Results at C30 (decode; distinct % / token-weighted %)
| W (tokens) | B8 (global 32) | B16 (global 64) | B64 (global 256) |
|---|---|---|---|
| 32 | 42.27 / 62.62 | 35.15 / 57.97 | 29.82 / 56.30 |
| 64 | 42.38 / 62.95 | 35.24 / 59.18 | 29.89 / 58.02 |
| **128** | **42.41 / 63.03** | **35.34 / 59.39** | **29.95 / 59.16** |
| 256 | 42.39 / 63.01 | 35.41 / 59.44 | 29.99 / 59.78 |
| 512 | 42.32 / 62.85 | 35.44 / 59.39 | 30.07 / 59.95 |
| 1024 | 42.34 / 62.84 | 35.43 / 59.16 | 30.11 / 59.95 |
| all-history | 41.86 / 60.92 | 35.30 / 57.34 | 30.20 / 57.83 |
| current-event | 42.28 / 62.64 | 35.27 / 59.22 | 30.01 / 59.90 |

- **Range across W from 32 to 1024 (distinct hit rate):** 0.14 pp (B8), 0.29 pp (B16), 0.29 pp (B64).
- **Seed noise:** BR-seed standard deviation is ≤0.007 pp, so these differences are systematic but small.
- **Gap from W=128 to the best W in the sweep:** 0 pp at B8, 0.10 pp at B16, 0.16 pp at B64.
- **Gap at B64 to all-history:** 0.25 pp.
- **Token-weighted hit rate** is more sensitive at large batch. At B64 it goes from 56.3 % (W32)
  to 59.2 % (W128) and 60.0 % (W ≥ 512).
- **Prediction (Spearman, h=4).** It rises with W and saturates near W=256:
  - B8: 0.746 → 0.768 (W128) → 0.775;
  - B16: 0.723 → 0.771 → 0.778;
  - B64: 0.717 → 0.779 → 0.804.

C40 and C60 show the same shape (see WINDOW_SENSITIVITY.csv). The W effect grows slightly with
capacity. B16 at C60 moves from 69.09 % (W32) to 69.97 % (W128) and 70.08 % (W512).

## Rank-order robustness (RANK_ORDER_ROBUSTNESS.csv)
- **At W=128,** the largest absolute change in distinct hit rate versus the identity order,
  over reversed, two random and rank-balanced orders and all three capacities, is:
  - 0.04 pp at B8;
  - 0.04 pp at B16;
  - 0.18 pp at B64 (C60; 0.016 pp at C30).

  The largest token-weighted change at W=128 is 0.64 pp (B64, rank-balanced, C60). At C30 it is
  0.44 pp (B64, reversed).
- **Over all W, orders and capacities,** the largest change is 0.49 pp distinct (B16, W32, C60,
  reversed). The largest token-weighted change is 2.4 pp (B64, W32, random order at C30).
  Rank-order bias matters only when W is much smaller than the global batch.
- **When W < global batch** (B64 at W ≤ 128), the window does see only the last ranks. For
  example, at B64 with W=128 the score is in effect an average over ranks 2 and 3 of the current
  event. In that regime both the full current-event mean and the rank-balanced window predict
  better: Spearman h=4 is 0.802 and 0.794 versus 0.779. The current-event mean gains over W=128
  by 0.07/0.13/0.23 pp distinct and 0.75/0.88/0.96 pp token-weighted at C30/C40/C60.

## Interpretation
- **W=128 is not uniquely optimal, but it sits on a near-flat plateau.** For distinct-expert hit
  rate, every W from 64 to 1024 lands within 0.3 pp at every batch and capacity tested. That is
  far smaller than the gap between gate-score and LFU or LRU in the history study (≥2 pp at B8,
  ≥18 pp at B16 and B64).
- **Best W by batch at C30.** At B8 (global 32), 128 is the best value. At B16 and B64 a larger
  window (256–1024, or all-history at B64) is slightly better, by 0.1–0.25 pp distinct and up to
  0.8 pp token-weighted at B64.
- **The two extremes are worse.** A very short window (W=32, or current-only at small batch) and
  pure cumulative history at small batch (B8: −0.55 pp distinct, −2.1 pp token-weighted) both
  lose. So some recency plus some smoothing helps, but the exact value inside 64–1024 matters
  little.
- **Defensible paper statement:** "W=128 is a fixed, untuned choice. Replay hit rates are
  insensitive (≤0.3 pp) to W between 64 and 1024 tokens. Windows at least as large as the global
  batch slightly improve prediction at B64." Any TPOT impact is unmeasured. At these magnitudes,
  any such impact is expected to be small, but that has not been verified.

## Provenance
- **Code base:** git a0d77a9 (paper_result), worktree `/home/hwlee/mgo-ferry-appendix`, branch
  `claude/ferry-appendix-sensitivity-20261011`, with the uncommitted additions listed below. The
  capture `status.json` records `source_commit` as a0d77a9.
- **GPUs:** physical 0,1,4,5 only. Each capture ran alone and serially (B8 → B16 → B64,
  352/361/407 s). Before each capture nvidia-smi showed no compute apps. No background load was
  restored (`restored: []`), and `remaining_gpu_occupants` was `[]` after every run. Worker CPU
  affinity came from `timing_stability_numa_20261004/topology.json`, as in the headline worker.
- **Capture command,** with b ∈ {8,16,64}:
  ```
  MGO_HEADLINE_ROOT=/home/hwlee/mgo-results/ferry_appendix_window_20261011 MGO_FERRY_WINDOW_PROBS=1 \
  /home/hwlee/sub-moe/phase01/.venv/bin/python scripts/run_headline_job.py --label window_B{b}_L256_H256_v1 \
   --system Ours --worker headline_ours_worker.py --cell R4_C30_B{b}_L256_O257 --repeats 1 --policy BR \
   --prefill-optimized --prefill-layout-fast --legacy-decode-layout --capture-eviction-trace \
   --workloads experiments/main_eviction_history_20261007/WORKLOADS.json
  ```
- **CPU sweep,** about 72 s on 24 processes pinned to cores 96–127:
  ```
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=<cpu_deps>:mgo_v2:mgo_v2/scripts python scripts/ferry_window_sensitivity.py \
   --capture <R>/window_B{8,16,64}_L256_H256_v1 --old /home/hwlee/mgo-results/headline_r4_20261007/main_eviction_B{8,16,64}_L256_H256_v1 \
   --output experiments/ferry_appendix_sensitivity_20261011/window --workers 24
  CUDA_VISIBLE_DEVICES="" ... python scripts/ferry_window_seed_noise.py <out> <R>/window_B{8,16,64}_L256_H256_v1
  python scripts/ferry_window_figure.py <out>
  ```
- **Raw files** are in `/home/hwlee/mgo-results/ferry_appendix_window_20261011/window_B{8,16,64}_L256_H256_v1/`.
  Per-rank `window_probs_rank*.npz` SHA256 values are in VALIDITY.json and `window_probs_rank*.json`.
- **Files in this directory:**
  - WINDOW_SENSITIVITY.csv
  - RANK_ORDER_ROBUSTNESS.csv
  - SEED_NOISE.csv
  - ALL_ROWS.json
  - VALIDITY.json
  - FIG_WINDOW_SWEEP.pdf/png
