# FERRY Appendix C: why Gate Score, and how to choose W (2026-10-13)

Status: complete. Nothing is committed. Worktree `/home/hwlee/mgo-ferry-appendix`, base a0d77a9.

## Setup

- **Model and data.** Qwen3-30B-A3B, R4 on physical GPUs 0/1/4/5, ShareGPT.
  - Input 512 uses the main-table `main_table_2x2_20261008` manifests.
  - B8 is the first 32 requests of the B16 manifest.
- **Cache and admission.** C30 (461/461/461/460 slots) with BR seed 42; C20/C40/C60 are CPU counterfactuals on the same traces. Prefetch is OFF.
- **Traces.**
  - Three new raw router-probability captures at L512 for B8/B16/B64 (`MGO_FERRY_WINDOW_PROBS=1`, 256 decode forwards).
  - Replays use the first 64 forwards, which is the paper's O64 horizon, and also the full 256.
  - The 2026-10-11 L256 captures were re-analysed as a robustness check.
- **Only Step 4 changes.** Every policy uses the same admission, the same pinning of active experts, and the same victim order (score, then last-use, then key). The policies differ only in the score.
- **Policies.** All windowed scores use a W-token rolling window per layer built with the production `GateHistory`.
  - **Full Gate (FERRY):** mean router probability over all experts.
  - **Selected-only Gate:** probability kept only for each token's top-8 experts.
  - **Selection Count:** windowed top-8 indicator.
  - **Baselines:** LRU and LFU-cumulative (distinct events).
  - **References:** random, the Belady oracle (OPT-BR), and a diagnostic *Selection Count + gate tie-break*.
- **Parity.**
  - W=128 Full Gate replay equals the live `reference_metrics` and `final_slots` bit-exactly on all 6 captures (L256 and L512).
  - The B16/B64 L512 captures ran after the live opt-in edits below, so this parity also confirms that the default path is unchanged.
  - The logging kernel is asserted equal to `ferry_eviction_sensitivity.replay2` in every cell.
- **Live TPOT.** 28 runs, all PASS.
  - Main-table Ours profile (`--ours-final`: LA_CA_NEAR, native, prefetch OFF), L512/O64, 3 target repeats, physical GPUs 0/1/4/5, serial.
  - Policies are set by `MGO_FERRY_EVICT_SCORE={full,selected,count,lru}` and the window by `MGO_FERRY_GATE_WINDOW`. LRU feeds zero gates, so the existing last-use tie-break makes it exactly LRU.
  - The baseline run reproduces the main-table H2D count exactly (179,577 at B16 C30).

## 1. Policy comparison (Fig. C.1; `L512/TABLES_H64.md`)

Decode distinct hit %, L512, 64 decode steps:

| Policy | B8 C30 | B16 C30 | B64 C30 | B8 C60 | B16 C60 | B64 C60 |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 0.0 | 0.0 | 0.0 | 35.2 | 2.7 | 0.0 |
| LFU-cumulative | 34.7 | 3.5 | 0.0 | 83.2 | 75.6 | 41.8 |
| random | 15.4 | 10.1 | 5.7 | 62.8 | 52.1 | 38.9 |
| Selection Count | 44.7 | 37.7 | 30.6 | 83.3 | 74.9 | 61.8 |
| Selected-only Gate | 45.8 | 38.9 | 31.5 | 84.3 | 77.4 | 65.4 |
| **Full Gate (FERRY)** | 45.0 | 38.8 | 31.7 | 80.0 | 74.6 | 64.9 |
| Belady oracle | 48.2 | 40.0 | 32.6 | 90.7 | 81.3 | 66.9 |

- **Token-weighted hit rate at C30** is 65.8/67.5/67.3 % for Full Gate, against 68.6/70.3/69.9 % for Selected-only and 68.1/69.8/69.4 % for Selection Count.
- **Re-fetches** are ≥99.99 % of H2D fetches for every policy, because prefill admits almost every expert first. Re-fetch count therefore does not separate the policies; report H2D fetches instead.
- **Robustness.** The ordering is the same at L256, at 256 decode steps, and at C20/C40.

### Live TPOT (Fig. `TPOT/FIG_C1_TPOT`, `TPOT/TPOT_TABLE.md`)

TPOT median of 3 repeats in seconds; relative change vs Full Gate in parentheses:

| | B8 C30 | B16 C30 | B64 C30 | B8 C60 | B16 C60 | B64 C60 |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 0.4545 (+4.3%) | 0.5071 (+2.2%) | 0.5968 (+4.2%) | 0.4424 (+5.5%) | 0.5136 (+9.7%) | 0.6062 (+7.2%) |
| Selection Count | 0.4400 (+1.0%) | 0.4893 (−1.4%) | 0.5848 (+2.1%) | 0.4122 (−1.7%) | 0.4733 (+1.1%) | 0.5748 (+1.7%) |
| Selected-only | 0.4345 (−0.3%) | 0.4882 (−1.6%) | 0.5795 (+1.2%) | 0.4169 (−0.6%) | 0.4766 (+1.8%) | 0.5683 (+0.5%) |
| **Full Gate** | 0.4356 | 0.4961 | 0.5728 | 0.4192 | 0.4680 | 0.5653 |

- **LRU is consistently slower**, by +2–10 %. Its live H2D fetch count is 1.6–3.6x that of Full Gate at C30 and up to 3.9x at C60.
- **The gate-family policies are within ±2 % of each other.** The TPOT order does not follow the fetch order. For example, at C60 B16 Selected-only fetches 10 % fewer experts yet is 1.8 % slower. Most H2D is overlapped, so fetch differences of this size do not reach TPOT.
- **Live fetch order matches the replay.** Selected-only fetches fewer than Full Gate at B8/B16 and at every C60 cell; Full Gate fetches fewest at B64 C30; Selection Count fetches more than Full Gate at C30.
- **Defensible claim:** the router-probability scores avoid LRU's +2–10 % TPOT loss. No TPOT ranking among Full, Selected-only and Count is supported. There is one run per cell; within-run repeat spread is ≤4 %, and cross-run drift is not measured.

## 2. Why Gate Score beats the baselines (Fig. C.2)

1. **LRU fails by scanning.** Layers run in order, so the LRU victim is always an expert of the layer about to execute. It stays at 0 % until the budget approaches the per-step working set (65–98 % of experts).
2. **LFU fails by saturating at large batch.** With many tokens per step, almost every expert is active in every step, so distinct-event counts stop discriminating: LFU is 0 % at B64 C30. Router probabilities are continuous, so they keep discriminating.
3. **Full Gate beats Selection Count through tie resolution, not extra information.**
   - At W=128, Selection Count has a tied minimum score in 49–63 % of evictions (Fig. C.2a), and those ties fall through to the LRU last-use order.
   - Breaking only the ties with the full gate (*Count + gate tie-break*) recovers to Full Gate or above at every W, batch and budget (Fig. C.2b).
   - Full Gate essentially never ties.
4. **The "near-selected expert" claim (old C.1) is not supported.**
   - 64 % of Qwen3 router mass lies outside the top-8, and the 9th probability (0.022) is close to the 8th (0.025).
   - However, experts unselected in the window make up only 3–9 % of next-step activations.
   - Full Gate ranks them better than chance (AUC 0.63–0.64), but the out-of-top-k mass dilutes the ranking of the experts that matter.
   - As a result, Selected-only ≥ Full on distinct hits at B8/B16 and at every C60 cell, and +3 pp on token-weighted hits everywhere.
   - The gap grows with budget (Fig. C.2c): Selected-only − Full = +0.1–0.8 pp at C30 and +0.5–4.3 pp at C60.
5. **Predictiveness (optional Fig. C.2 of the original plan) contradicts the claim.** At W=128 and h=4, the Spearman correlation with future demand is 0.71–0.79 for Full Gate, against 0.85–0.92 for Selected-only and Selection Count. Do not include this as support.

## 3. Window W (Table C.1, Fig. W)

Full Gate, L512, 64 decode steps, distinct hit %:

| W | 8 | 16 | 32 | 64 | **128** | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B8 C30 | 43.58 | 43.92 | 44.81 | 44.87 | **44.95** | 44.91 | 44.74 | 44.33 |
| B16 C30 | 38.05 | 38.29 | 38.50 | 38.71 | **38.80** | 38.78 | 38.73 | 38.65 |
| B64 C30 | 31.39 | 31.49 | 31.60 | 31.62 | **31.67** | 31.69 | 31.70 | 31.68 |
| B8 C60 | 76.46 | 77.44 | 79.83 | 79.98 | **79.98** | 79.87 | 79.36 | 78.70 |
| B16 C60 | 72.36 | 73.03 | 73.69 | 74.49 | **74.58** | 74.52 | 74.40 | 74.22 |
| B64 C60 | 64.21 | 64.35 | 64.74 | 64.80 | **64.87** | 65.00 | 65.00 | 64.81 |

- **Plateau.** W∈[64,256] is within 0.15 pp of the best value in every cell. W=128 is the best or within 0.13 pp of it.
- **Small windows lose.** Below about half the global batch per step (W/global batch < 1/2), up to 3.5 pp is lost.
- **Very long windows lose at small batch** (B8 W=1024: −0.6 pp at C30, −1.3 pp at C60).
- **Selection Count needs W ≥ 512** to catch up, because coarse counts tie. Robustness at small W is a genuine advantage of continuous scores.
- **Live TPOT across W** (Full Gate, C30) is W32/W128/W512 = 0.4877/0.4961/0.4949 s at B16 and 0.5709/0.5728/0.5950 s at B64. Fetch counts differ by ≤0.4 %. The +3.9 % at B64 W512 does not follow from the fetches; it is a single run and is not interpreted.

**Recommendation:** keep W=128. Paper wording: "W=128 tokens lies on a plateau (W∈[64,256] within 0.15 pp); it must cover at least about half a decode step of global tokens."

## Suggested Appendix C wording

- **Keep:** Gate Score is far better than LRU, LFU and random, and within 1–3.5 pp of Belady at C30 (Belady − Full Gate = 3.2 / 1.2 / 0.9 pp at B8 / B16 / B64).
- **Rewrite C.1:** the advantage comes from a continuous, router-weighted windowed frequency. It has no ties and stays robust at small W, which selection counts do not.
- **Drop:** the claim that unselected experts' probability is the key ingredient, and the Spearman figure.
- **Note the budget dependence:** at large budgets (C60) Selected-only Gate is 0.5–4.3 pp better on hit rate, with no consistent TPOT difference.

## Files

- **Scripts** (new, under `mgo_v2/scripts/`):
  - `ferry_gate_policy_analysis.py` (CPU replay)
  - `ferry_gate_policy_figures.py`
  - `ferry_gate_tpot_summary.py`
- **Live opt-in edits** (default unchanged, parity verified):
  - `mgo_v2/eviction.py` (`ferry_score_rows`, `FERRY_GATE_WINDOW`)
  - `mgo_v2/live_metadata.py`
  - `mgo_v2/decode_runtime.py`
  - `examples/headline_ours_worker.py`
- **Outputs:**
  - `L512/`, `L256/`: `POLICY_ROWS.csv`, `TABLES_H{64,256}.md`, `FIG_C1_*`, `FIG_C2_*`, `FIG_W_*`
  - `TPOT/`: `TPOT_ROWS.csv`, `TPOT_TABLE.md`, `FIG_C1_TPOT`
  - `workloads/`
  - Runners: `run_captures.sh`, `run_tpot.sh`
- **Raw data:** `/home/hwlee/mgo-results/ferry_gate_score_20261013/` (captures and `tpot/`).
- **CPU commands** (from `mgo_v2`):
  - Environment: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=<cpu_deps>:.:scripts`
  - Replay: `taskset -c 48-95 python scripts/ferry_gate_policy_analysis.py --capture <3 dirs> --tag L512 L512 L512 --output experiments/ferry_gate_score_20261013/L512 --workers 24` (202 s)
  - Figures: `python scripts/ferry_gate_policy_figures.py experiments/ferry_gate_score_20261013/L512 64`

## Caveats

- Qwen3 only. DeepSeek-V2-Lite was not repeated in this packet; the 10-11 DeepSeek replay shows the same LRU/LFU pattern, but it has no raw probabilities for the Selected-only and Selection Count variants.
- There is one frozen trace per batch and one BR seed. Hit rates are replay counts under BR admission; the live runs use LA_CA_NEAR admission.
- C60 is a CPU counterfactual on the C30-captured traces. Its live TPOT runs are real.
- TPOT has one run per cell, each with 3 repeats.
