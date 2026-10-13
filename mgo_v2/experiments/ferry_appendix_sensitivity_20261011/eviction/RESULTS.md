# FERRY appendix: eviction-policy sensitivity (cache budget x batch)

Date 2026-10-11. CPU-only replay; `CUDA_VISIBLE_DEVICES=""` for every command, no GPU touched.
Files: `EVICTION_SENSITIVITY.csv` (all 608 cells), `FIG_EVICTION_SENSITIVITY.{pdf,png}`,
`PROVENANCE.json` (hashes, per-cell parity checks, working-set statistics).

## 1. Question

At C30 the earlier study (`main_eviction_history_20261007`) found LRU and LFU-reset
at exactly 0% decode hit. That result is correct, but on its own it reads like a strawman. This sweep varies the
MAIN budget from 10% to 100% of all experts and covers three batch sizes. It also
adds token-weighted hit rate, a random baseline and Belady oracles. The goal is to
show where recency/frequency policies recover and how close gate-score is to optimal.

## 2. Workloads (frozen traces, not re-captured)

| Source | Model | Layers x experts | Local B (global) | Decode steps | Trace receipt |
|---|---|---|---|---|---|
| q8/q16/q64 | Qwen3-30B-A3B, R4, input 256 | 48 x 128 = 6144 | 8/16/64 (32/64/256) | 256 (+1 prefill) | `trace_receipt.json` status PASS, prefetch false, all-rank agreement, content sha256 re-verified |
| ds16 | DeepSeek-V2-Lite, R4, ShareGPT input 512 | 26 x 64 = 1664 | 16 (64) | 63 (+1 prefill) | per-rank route sha256 equals the D1 `repeat1_rank*.json` `route_sha256`, status PASS |

The Qwen traces are `/home/hwlee/mgo-results/headline_r4_20261007/main_eviction_B{8,16,64}_L256_H256_v1/routing_trace.npz`.
DeepSeek is repacked with the unchanged `replay_deepseek_eviction_hit.pack()`
(rolling 128-row router-probability gate vector). There are no other DeepSeek
`routing_trace.npz` files, and no other DeepSeek traces were used.

## 3. Method and exact semantics

The decision kernel is copied from `main_eviction_history_replay.replay` and
extended. Only `scripts/ferry_eviction_sensitivity.py` was added; the original
kernel was not modified.

- **Event:** one (step, layer) forward. An expert is *active* if `counts>0`. A
  *distinct request* is one (decode step, layer, expert).
- **Hit:** the active expert is resident in MAIN on any rank before admission.
  Every miss is admitted (demand paging). There is no prefetch, substitution or
  replication.
- **Admission:** BR quota, the same `np.random` shuffle, seed 42, 4 ranks. A miss
  can evict only a slot on its assigned rank.
- **Pinning:** experts active in the current event are never victims. This
  applies to every policy.
- **Victim selection:** the slot with minimum `score`. Ties go to the smaller
  last-use, then the smaller key. This order is identical for all policies.
  - gate-score: the current W128 gate-score of the victim. The gate vector for a
    layer is refreshed when that layer's event runs.
  - **LFU** (key = (layer, expert)): +1 per distinct (step, layer, expert)
    activity event, independent of token count.
    - *LFU-reset* zeroes the key's frequency and last-use when it is evicted.
    - *LFU-cumulative* keeps them across eviction and re-admission.
  - **LRU** (key = (layer, expert)): score 0, so the victim is the minimum
    last-use event index.
    - *Reset* and *cumulative* are asserted bit-identical in every cell. A
      re-admitted key is pinned until its last-use is refreshed.
  - **random:** a uniform score from a separate xorshift64 stream (seed 1234),
    so the BR `np.random` stream is untouched. Empty slots are still filled
    first.
  - **OPT-BR (oracle):** Belady victim selection (farthest next use of that
    (layer, expert)) with the same BR rank admission. This is a strong oracle
    reference, not a proven bound, because BR fixes which rank a miss lands on.
  - **OPT-pooled (oracle):** Belady on one pooled cache with the same total slot
    count. This is an upper bound on distinct hits for any demand-admission
    policy with that budget. Empirically, no policy exceeds it in any of the 152
    budget cells.
- **Budget:** `total = floor(L*E*pct/100)`, split across ranks as `total//4 +
  (r < total%4)`, as in the receipts (C30 Qwen = 461/461/461/460 = 1843).
  DeepSeek subtracts 2 reserved non-MAIN slots per rank, as the prior DeepSeek
  study did (C30 = 123/123/123/122). As a result, DeepSeek at 100% has 1656 slots
  for 1664 experts.
- **Metrics:** decode only; prefill step 0 is excluded from numerators and
  denominators.
  - `hit_rate = decode_hit / (decode_hit + decode_miss)`. The denominator is
    distinct decode requests: q8 1,024,059; q16 1,252,988; q64 1,476,811;
    ds16 102,818.
  - `token_weighted_hit_rate = sum(counts of hit experts) / sum(counts)`. The
    denominator is the number of decode token-to-expert assignments, equal to
    global batch x top-k x layers x steps: q8 3,145,728; q16 6,291,456;
    q64 25,165,824; ds16 629,856.
  - `evictions` counts residents removed.
  - `reloads` counts misses on a key that was admitted earlier, including in
    prefill.
- **Feasibility:** a cell is infeasible if BR cannot place a miss because every
  slot on its rank is pinned. That case is caught and recorded, not raised. No
  cell was infeasible, not even 10% at B64, where the maximum per-layer active
  count is 128.

## 4. Parity checks (all PASS; listed per cell in `PROVENANCE.json`)

1. **Reference kernel:** for gate-score, LFU-reset, LFU-cumulative, LRU-reset and
   LRU-cumulative, per-event hit, miss, eviction and reload, plus final slots,
   are bit-identical to the unmodified `main_eviction_history_replay.replay`.
   This holds in all 76 budget x source cells, so the extension changed no
   behaviour.
2. **Source capture:** at the Qwen source capacity (C30, 461/461/461/460),
   gate-score matches the capture's `reference_metrics` and `final_slots`
   exactly for B8, B16 and B64.
   - At other budgets, gate-score has no physical reference by construction.
     Those cells are CPU counterfactuals.
3. **Prior DeepSeek results:** DeepSeek C30 and C60 reproduce all 5 policies in
   `deepseek_eviction_hit_20261009/RESULTS.json` exactly.
4. **Prior Qwen results:** Qwen C30 numbers equal `main_eviction_history_20261007/RESULTS.md`
   (for example, B8 gate-score 434,276 hits and 589,783 misses).
5. **Invariants:**
   - hit + miss equals the number of active experts in every event.
   - evictions ≤ misses and reloads ≤ misses.
   - Occupied slots equal misses minus evictions.
   - Token hits ≤ tokens.
   - LRU-reset equals LRU-cumulative.
   - OPT-pooled ≥ every policy.

## 5. Results

### 5.1 C30 (paper operating point), distinct / token-weighted decode hit %

|Model|B/rank|Slots|gate-score|LFU-cum|LFU-reset|LRU (both)|random|OPT-BR|OPT-pooled|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Qwen3|8|1843|42.4 / 63.0|40.1 / 61.7|0.0 / 0.0|0.0 / 0.0|13.1 / 14.4|44.9 / 49.6|44.9 / 49.4|
|Qwen3|16|1843|35.3 / 59.4|16.7 / 28.5|0.0 / 0.0|0.0 / 0.0|7.8 / 8.4|36.3 / 39.1|36.3 / 39.0|
|Qwen3|64|1843|29.9 / 59.2|0.0 / 0.0|0.0 / 0.0|0.0 / 0.0|4.7 / 4.9|30.5 / 31.4|30.5 / 31.4|
|DeepSeek|16|491|26.1 / 33.1|0.0 / 0.0|0.0 / 0.0|0.0 / 0.0|3.6 / 3.6|27.4 / 28.0|27.4 / 28.0|

Raw counts (hits, misses, evictions, reloads) are in the CSV.

### 5.2 Budget sweep (selected budgets; full 10-100% grid in the CSV and figure)

|Model|B/rank|Budget|Slots|gate-score|LFU-cum|LFU-reset|LRU|random|OPT-BR|OPT-pooled|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Qwen3|8|20%|1228|27.9 / 48.0|21.1 / 37.5|0.0 / 0.0|0.0 / 0.0|3.6 / 4.0|29.2 / 32.3|29.2 / 32.1|
|Qwen3|8|40%|2457|55.7 / 74.4|56.4 / 76.8|15.8 / 19.7|0.1 / 0.1|26.7 / 29.0|60.5 / 66.7|60.5 / 66.5|
|Qwen3|8|50%|3072|67.5 / 82.9|70.2 / 86.3|48.6 / 60.8|0.9 / 1.4|42.5 / 45.4|75.9 / 83.1|75.9 / 83.0|
|Qwen3|8|60%|3686|77.5 / 89.2|81.3 / 92.1|71.9 / 84.1|9.3 / 11.9|58.5 / 61.7|88.3 / 94.7|88.4 / 94.8|
|Qwen3|8|75%|4608|89.1 / 95.5|93.0 / 97.3|88.8 / 95.1|89.6 / 95.4|80.2 / 82.7|96.8 / 98.8|96.9 / 98.9|
|Qwen3|16|20%|1228|22.7 / 44.9|0.0 / 0.0|0.0 / 0.0|0.0 / 0.0|1.7 / 1.8|23.5 / 25.2|23.5 / 25.2|
|Qwen3|16|40%|2457|47.6 / 70.7|40.9 / 64.2|0.0 / 0.0|0.0 / 0.0|18.0 / 19.2|49.1 / 53.0|49.1 / 52.8|
|Qwen3|16|50%|3072|59.2 / 79.6|58.6 / 81.4|24.6 / 30.9|0.1 / 0.2|31.1 / 32.9|61.9 / 66.5|61.9 / 66.4|
|Qwen3|16|60%|3686|70.0 / 86.6|71.9 / 89.7|51.9 / 66.4|0.5 / 0.7|46.1 / 48.4|74.6 / 79.9|74.6 / 79.7|
|Qwen3|16|75%|4608|84.0 / 94.0|87.7 / 96.4|82.3 / 93.3|16.2 / 19.5|69.8 / 71.9|91.9 / 96.7|91.9 / 96.7|
|Qwen3|64|20%|1228|19.1 / 44.9|0.0 / 0.0|0.0 / 0.0|0.0 / 0.0|0.8 / 0.8|19.6 / 20.1|19.6 / 20.0|
|Qwen3|64|40%|2457|40.7 / 70.2|0.0 / 0.0|0.0 / 0.0|0.0 / 0.0|12.3 / 12.6|41.4 / 42.7|41.4 / 42.6|
|Qwen3|64|50%|3072|51.4 / 78.7|0.2 / 0.3|0.0 / 0.0|0.0 / 0.0|22.9 / 23.4|52.3 / 53.9|52.3 / 53.9|
|Qwen3|64|60%|3686|61.9 / 85.7|25.2 / 36.2|0.1 / 0.1|0.0 / 0.0|35.9 / 36.6|63.1 / 65.1|63.1 / 65.1|
|Qwen3|64|75%|4608|77.1 / 93.0|76.7 / 95.0|64.4 / 81.7|0.3 / 0.4|59.0 / 59.9|79.4 / 81.5|79.4 / 81.5|
|DeepSeek|16|20%|324|15.7 / 21.5|0.0 / 0.0|0.0 / 0.0|0.0 / 0.0|0.5 / 0.5|16.7 / 17.0|16.7 / 17.1|
|DeepSeek|16|40%|657|36.5 / 43.7|0.0 / 0.0|0.0 / 0.0|0.0 / 0.0|10.2 / 10.3|38.1 / 38.8|38.1 / 38.8|
|DeepSeek|16|50%|824|47.0 / 53.9|0.0 / 0.0|0.0 / 0.0|0.0 / 0.0|19.9 / 20.0|48.6 / 49.6|48.6 / 49.6|
|DeepSeek|16|60%|990|57.5 / 63.6|1.7 / 1.9|0.0 / 0.0|0.0 / 0.0|32.2 / 32.4|59.1 / 60.2|59.1 / 60.1|
|DeepSeek|16|75%|1240|73.1 / 77.9|33.8 / 38.4|0.5 / 0.4|0.0 / 0.0|54.5 / 54.7|74.9 / 76.1|74.9 / 76.1|

### 5.3 Crossovers (first budget, on the 5% grid, with a distinct hit rate ≥ 1% and ≥ 10%)

|Source|per-step working set (mean, % of experts)|LRU|LFU-reset|LFU-cumulative|random|LFU-cum ≥ gate (distinct), budgets < 100%|
|---|---:|---|---|---|---|---|
|Qwen B8|4000 (65%)|55% / 65%|40% / 40%|10% / 15%|15% / 30%|≥ 40%|
|Qwen B16|4894 (80%)|65% / 75%|45% / 50%|25% / 30%|20% / 35%|≥ 55%|
|Qwen B64|5769 (94%)|85% / 90%|65% / 65%|55% / 60%|25% / 40%|≥ 80%|
|DeepSeek B16|1632 (98%)|85% / 95%|80% / 80%|60% / 70%|25% / 40%|never|

The per-step working set is the number of distinct (layer, expert) pairs active
in one decode step, summed over all layers (`PROVENANCE.json` `working_set`).
Experts active in every one of the 256 decode steps: 419, 1176 and 3104 for
Qwen B8, B16 and B64, and 915 of 63 steps for DeepSeek.

## 6. Interpretation

1. **The zero for LRU is a scan pathology, and it persists across the realistic
   budget range.**
   - Layers are visited cyclically. The LRU victim is therefore always an expert
     of the layer that is about to run, i.e. the oldest-touched one.
   - LRU stays at 0–1% hit until the budget approaches the per-step working set:
     about 65% of all experts for B8, 80% for B16, 94% for B64 and 98% for
     DeepSeek. It then jumps sharply, for example from 9% to 84% between 60% and
     70% at B8.
   - Below that point LRU is worse than **random** at every budget. Random gets
     13% at B8 C30 and 4.7% at B64 C30.
   - LFU-reset forgets an expert on eviction, so it degenerates to LRU until
     experts survive long enough to accumulate counts. It recovers 15–25 points
     of budget earlier than LRU.
2. **LFU-cumulative is the strong conventional baseline and should be
   presented as such.** Its long-run frequency acts like a learned static hot
   set.
   - At B8 it is within 2.3 points of gate-score at C30 and *beats* gate-score
     from 40% budget upward. At 60% it reaches 81.3 vs 77.5 distinct and 92.1 vs
     89.2 token-weighted.
   - At B16 it beats gate-score from 55% upward; at B64 from 80% upward.
   - At large batch, distinct-event frequency saturates: 3104 experts are active
     at every step of B64. Frequency then stops discriminating, and LFU-cumulative
     collapses (0% up to 50% budget at B64).
   - Gate-score's continuous router probabilities keep that discrimination.
     Gate-score's advantage is therefore largest at **low budget and large
     batch**, which is the regime FERRY targets.
   - On DeepSeek, LFU-cumulative never reaches gate-score below 100%.
3. **Gate-score is close to the oracle for distinct hits.**
   - At C30 it is 2.5 / 1.0 / 0.6 / 1.3 points below OPT-pooled for Qwen B8 /
     B16 / B64 and DeepSeek.
   - The gap stays at 1–2 points up to about 50% budget and widens to about 4–11
     points at 60–75% for small batches. That is the same region where
     LFU-cumulative overtakes it.
   - OPT-BR is within 0.15 points of OPT-pooled everywhere, so the BR rank
     partition costs essentially nothing in hit rate.
4. **Token-weighted hit rate favours gate-score even more.**
   - Gate-score keeps the experts that carry the most router mass, so it hits
     59–63% of token assignments at Qwen C30. That is well above the 31–50% of
     the distinct-hit Belady oracle. Belady optimises distinct hits, not tokens,
     so it is **not** an upper bound on the token-weighted metric.
   - LFU (distinct-event) and LRU have no token awareness.
5. **Batch sensitivity:** for every policy the hit rate falls as batch grows,
   because the working set grows. The ranking gate-score ≳ LFU-cumulative ≫
   random > LFU-reset ≥ LRU is stable for budgets of 40% or less. It reorders
   only at high budget for small batch.

## 7. Caveats (state these in the appendix)

- **Replay hit rate is not TPOT.** These are counts of cache events, not timing.
  The effect of a hit on latency depends on H2D overlap, batching of misses and
  the critical rank. Only C30 gate-score has physical counterparts (the source
  runs).
- **One frozen trace per batch and model, one BR seed (42), one random seed.**
  There are no confidence intervals across prompts or seeds. DeepSeek covers
  only 63 decode steps at B16, with a C20-captured route.
- **Is the trace policy-independent?**
  - The source traces were captured under gate-score eviction. The receipts and
    the PLAN show prefetch OFF, no substitution and no replication, with exact
    BF16 expert execution.
  - Routing (top-k ids, counts) and the router-probability gate vectors depend
    only on hidden states. With exact execution, the hidden states do not depend
    on which experts are resident.
  - Residual dependence: the owner map can change which rank computes an expert,
    and with it the reduction order. That is a possible BF16 nondeterminism, and
    it is not separately controlled (the PLAN notes routing "may depend on its
    BF16 arithmetic").
  - Gate-score reproducing the source at C30 is a parity check of the replay. It
    is not an advantage given to gate-score, because the other policies replay
    the identical counts and gates.
- Counterfactual budgets (≠ C30 Qwen; all DeepSeek budgets) have no physical
  validation.
- **Scope of LFU and LRU:**
  - LFU here is per-expert, distinct-event LFU as requested. Token-weighted LFU,
    aging LFU, per-layer quotas and layer-aware LRU were **not** evaluated. A
    per-layer partitioned LRU would avoid the cross-layer scan and is the
    natural reviewer counter-proposal.
  - The DeepSeek budget reserves 2 slots per rank, so its "100%" is 1656/1664
    slots.

## 8. Provenance and reproduction

- Scripts (new, untracked):
  - `mgo_v2/scripts/ferry_eviction_sensitivity.py`; its sha256 is in `PROVENANCE.json`.
  - `mgo_v2/scripts/ferry_eviction_sensitivity_plot.py`.
- Imported unchanged:
  - `main_eviction_history_replay.py` (reference kernel).
  - `replay_deepseek_eviction_hit.py` (DeepSeek packer).
- Trace file sha256:

  | Trace | File sha256 | Content sha256 (counts‖gates‖ref) |
  |---|---|---|
  | B8 | 2bc2d9ee…2a63 | cea5f091…ead8e |
  | B16 | 4f5dfa4b…d415 | 968cde90…6427 |
  | B64 | 30f110b5…bca0 | 91a643a0…1c37 |

  The DeepSeek route files are 130f7251…, 741f9af8…, 4efac876… and 0f29858….
  Full hashes are in `PROVENANCE.json`.
- Commands (from `/home/hwlee/mgo-ferry-appendix/mgo_v2`):

```sh
export CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=<scratch>/numba \
  PYTHONPATH=/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps
/home/hwlee/sub-moe/phase01/.venv/bin/python scripts/ferry_eviction_sensitivity.py --jobs 24
/home/hwlee/sub-moe/phase01/.venv/bin/python scripts/ferry_eviction_sensitivity_plot.py
```

The run used 24 CPU processes and took 112 s of wall time (about 25 CPU-min).
All seeds are fixed (BR 42, random 1234), so the run is deterministic by construction.
