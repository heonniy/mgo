# E5 / D1: metadata composition, payload, timing and parity (FERRY appendix)

Status: **CPU/code analysis only**, 2026-10-11. No GPU was used: every Python run had `CUDA_VISIBLE_DEVICES=""`. Timing numbers come from existing receipts under `/home/hwlee/mgo-results` and were not re-measured. Code ref `a0d77a9`, worktree branch `claude/ferry-appendix-sensitivity-20261011`. Nothing is committed.

Files in this directory:
- `METADATA_FIELDS.md`: the field table (local / exchanged / derived-replicated / offline), with file:line evidence.
- `METADATA_BYTES.csv`: decode bytes. `METADATA_BYTES_PREFILL.csv`: prefill bytes.
- `FIG_METADATA_BYTES.{pdf,png}`: the figure.
- `METADATA_TIME.csv`: per run × rank × phase. `METADATA_TIME_SUMMARY.csv`: the cross-rank view.
- `PARITY_CHECKS.json`: summary. `PARITY_CHECKS_RAW.json`: all 646 per-rank CPU checks.
- `scripts/`: the generators. `metadata_bytes.py`, `harvest_metadata_time.py` and `cpu_metadata_parity.py` are new. No existing source was modified.

## 1. Field table (summary; full table in METADATA_FIELDS.md)

- **Exchanged on each MoE-layer decode event** (one `all_gather_into_tensor`, `live_metadata.py:27`):
  - a 24-B int64 event header;
  - uint8 Top-K IDs [B,k];
  - (live path) float32 router-probability rows for the last min(B,128) tokens of each rank.
- **Local only:** the Top-K combine weights. The controller receives constant `ones`, so routing weights are not exchanged (`live_metadata.py:21,36`).
- **Derived identically on every rank, never sent:**
  - demand d_{r,e} and the required set;
  - residency C_r (`slots`/`owner`/`primary`);
  - evictable slot choice u_r (implicit in `choose_slot`, not stored);
  - load N_r (`estimate_loads`);
  - gate history H (W=128) and gate scores;
  - misses F, balanced quotas q_r, Near assignment, evictions/fetch list, dispatch destinations.
- **Offline:**
  - expert size S;
  - bandwidth groups / quota LUT B_g. These are used **only by R8 PCIe-quota policies**. The R4 main cells (policy 7, Near) use the balanced q_r and never read B_g.
- **Prefill** uses a different, padded path (`runtime.py:40-98`): int64 IDs, BF16 weights, and a float32 probability tail limited to the global last 128 rows.
- **Caveat:** the paper names `GlobalExpertController`. The main-cell runtime instead implements the same replicated design through `DecodePrefetchController` → numba `Policy` (Qwen), or `Policy` directly (DeepSeek). The appendix should cite the design, not that class, as the executed code.

## 2. Payload bytes (verified formulas)

Both formulas are confirmed from the code and checked on CPU by instantiating the unmodified classes (`record_bytes == send.numel()`, and the bytes handed to the all-gather equal the record; PARITY_CHECKS A):
- compact `24 + B·k + 4E` (`compact_metadata.py:11-12`);
- live `24 + B·k + 4E·min(B,128)` (`live_metadata.py:16`).

Model geometry, from `/home/hwlee/model/*/config.json`:

| Model | k | E | MoE layers | Expert size S |
|---|---|---|---|---|
| Qwen3-30B-A3B | 8 | 128 | 48 (`mlp_only_layers=[]`) | 3·2048·768·2 = 9,437,184 B |
| DeepSeek-V2-Lite | 6 | 64 | 26 (27 layers, `first_k_dense_replace=1`) | 3·2048·1408·2 = 17,301,504 B |

R4 decode, **live path (the path the main cells used)**:

| Model | B | per-rank send / event | all-gathered (R×) / event | per decode step (× L) | step / S |
|---|---|---|---|---|---|
| Qwen | 16 | 8,344 B | 33,376 B (32.6 KiB) | 1,602,048 B (1.53 MiB) | 0.170 |
| Qwen | 64 | 33,304 B | 133,216 B (130.1 KiB) | 6,394,368 B (6.10 MiB) | 0.678 |
| DeepSeek | 16 | 4,216 B | 16,864 B (16.5 KiB) | 438,464 B (0.42 MiB) | 0.025 |
| DeepSeek | 64 | 16,792 B | 67,168 B (65.6 KiB) | 1,746,368 B (1.67 MiB) | 0.101 |

Compact path, for comparison:
- Qwen: B16 664 / 2,656 B; B64 1,048 / 4,192 B (per rank / gathered per event).
- DeepSeek: B16 376 / 1,504 B; B64 664 / 2,656 B.

Per-event gathered metadata is 0.35% (Qwen B16) to 1.4% (Qwen B64) of a single expert copy.

"Gathered" means the R×record buffer that every rank ends up holding. Each rank receives (R−1)× the record from peers. NCCL protocol and ring overhead are not included.

Prefill (once per MoE layer, `METADATA_BYTES_PREFILL.csv`): the int64 IDs dominate. For Qwen B16/L512 this is 720,904 B per rank per layer. That figure matches the diagnostic receipt's `metadata_padded_send_bytes` exactly, apart from the 8-B count.

## 3. Which path the main-paper OURS cells used: **live** (both models)

Evidence:
1. **Selected attempts.** The selected attempt of all 16 `main_OURS` cells in `experiments/main_table_2x2_20261008/PROGRESS.json` resolves to a `status.json` under `/home/hwlee/mgo-results/headline_r4_20261007/mt2_*_ours_r3_*`.
2. **Qwen worker.** The 8 Qwen cells launched `examples/headline_ours_worker.py ... --policy LA_CA_NEAR` (native, prefetch-off, prefill-optimized, decode-layout-fast).
   - At each recorded `source_commit`, that worker sets `a.live_routes=True` unconditionally (`headline_ours_worker.py:71`).
   - It re-creates `LiveMetadata` on every reset and repeat (`:111`, `:137`).
   - `DecodeOffloadRuntime` selects `LiveMetadata` when `live_routes` is set (`decode_runtime.py:29-33`). Otherwise it falls back to `CompactMetadata` (`:35`).
3. **DeepSeek worker.** The 8 DeepSeek cells launched `examples/headline_ours_deepseek_worker.py`. At each source commit it constructs `LiveMetadata(self.batch, self.history, TOPK=6, EXPERTS=64, LAYERS=26)` and has no compact import.
4. **Byte formula at run time.** `live_metadata.py` at every one of the 16 source commits has `tail=min(batch,128)` and the same record formula. The Qwen B16/L512 commit `1a1ab118` is the Qwen-hard-coded k=8/E=128 version, which gives identical bytes.
5. **No compact evidence.** No main-cell worker imports `CompactMetadata`. The compact path survives only in older frozen-trace workers (`refactor_baseline_worker.py`) and checks.

## 4. Sensitivity: bytes vs local batch and the min(B,128) cap (FIG_METADATA_BYTES)

**Live payload size.**
- The live payload is dominated by probability rows.
- It grows linearly in B up to B=128 and is then flat. At B=256 only the IDs keep growing, by B·k bytes.
- Compact stays ≤ 10 KiB per event for every B ≤ 256.

**Why the cap at 128.** The gate history keeps the last W=128 rows of the global rank-ordered token stream (`eviction.py:42-48`). No rank therefore ever needs to contribute more than 128 rows. The cap is exactly lossless: GateHistory sums are identical with and without truncation (CPU check `tail_cap_min_B_128_lossless_vs_full_rows`).

**Redundancy above R·B = 128.** The cap is per rank, not global. Once R·B > 128, earlier ranks' rows are sent and then discarded:

| Local B | rows sent | rows used | redundancy |
|---|---|---|---|
| 16 | 64 | 64 | 1× |
| 64 | 256 | 128 | 2× |
| ≥128 | 512 | 128 | 4× |

**Unexploited saving.** A suffix-aware variant (as prefill already does, `runtime.py:73-80`) would reach the same exact history with about half (B64) to a quarter (B≥128) of the probability bytes. The fixed-size `all_gather_into_tensor` record makes that less convenient. This is a saving the main cells did not use, not a correctness issue.

**Where the cost scales.** Even uncapped, the payload stays ≤ 1.4% of one expert per event. Measured all-gather latency does not scale with these sizes (§5), so payload size is not the cost driver at R4 on NVSwitch.

## 5. Measured metadata time (harvested; METADATA_TIME*.csv)

**Data exists.** Phase spans from `LiveMetadata.collect_profiled` are in 10 existing post-generation diagnostic runs:
- Qwen ShareGPT R4/C30/L512, LA_CA_NEAR, GPUs 0,1,4,5;
- 63 decode steps × 48 layers = 3,024 events per rank;
- run dirs are listed in `scripts/harvest_metadata_time.py`.

Six of them match the main-cell configuration (serial native, no grouped mode): B8 ×1, B16 ×3, B64 ×2. The rest are grouped/compiled-dense variants.

**No DeepSeek metadata timing exists.** The DeepSeek worker calls `collect`, not `collect_profiled`.

Main-config runs, per-event medians across runs and ranks (current-stream spans, µs):

| phase | B8 | B16 | B64 |
|---|---|---|---|
| GPU packet pack | 139–145 | 137–149 | 141–153 |
| all-gather, rank with **least** wait (cross-rank min per event) | 107 (p95 127) | 106–110 (p95 125–132) | 110–111 (p95 129–130) |
| all-gather, rank with most wait | 928 | 966–1092 | 1107–1166 |
| D2H copy | 29–31 | 29–32 | 32–35 |
| CPU parse | 68–70 | 68–74 | 75–83 |
| gate-history update | 90–91 | 122–202* | 162–192 |
| demand histogram | 28–29 | 29–30 | 32–33 |
| instrumentation glue (`moe.metadata` residual) | 102–109 | 100–115 | 101–116 |

\* 202 µs is the older-commit run `expert_grouping_ablation/r4_a_diagnostic_v1`, from before the compiled gate-history update.

**Fraction of TPOT** (`METADATA_TIME_SUMMARY.csv`):

| | B8 | B16 | B64 |
|---|---|---|---|
| Lower-bound metadata cost per step | 29.0 ms | 31.0–35.1 ms | 35.4–35.6 ms |
| Diagnostic step time | 490 ms | 549–558 ms | 640–648 ms |
| Fraction | 5.9% | 5.6–6.3% | 5.5% |

How the lower bound is built:
- per event, take the cross-rank minimum all-gather span plus the slowest rank's non-collective metadata;
- sum over 48 layers;
- divide by the median token interval of the diagnostic generation.

Per-rank exclusive metadata spans, which include peer waiting inside the all-gather, are 7–13% of the step.

How to read this:
- The minimal all-gather span (~107–111 µs) is flat across B=8→64. That is a 4× growth in payload (16.7→133 KiB gathered), so the collective is latency- and launch-bound, not bandwidth-bound.
- The large and rank-dependent all-gather spans are mostly **arrival skew**: time spent waiting for the slowest rank's preceding layer work. That is exposed by the collective, not caused by metadata transfer.
- The non-collective cost (~0.47–0.59 ms per event, ×48) is mostly fixed per-event host and launch overhead. Instrumentation glue alone is ~0.1 ms per event.

**Caveats:**
- These are diagnostic passes with extra CUDA-event marks. Diagnostic step time is about 6–13% above the job's target TPOT (e.g. 549 vs 490 ms at B16), and the absolute ms include instrumentation overhead.
- They are spans on the current stream. Per AGENTS.md, collective spans are not pure NCCL wire time.
- There is one diagnostic run per configuration (B16 has three). There are no repeats of the diagnostic itself.

**Command to fill the gaps** (not run). This is a fresh Qwen measurement on the current tree. DeepSeek needs a `collect_profiled` hook added to `headline_ours_deepseek_worker.py` first, because its `DIAG` mode does not split metadata.

```
cd /home/hwlee/mgo-ferry-appendix/mgo_v2 && /home/hwlee/sub-moe/phase01/.venv/bin/python scripts/run_headline_job.py \
  --label e5_metadata_qwen_b16_diag --system Ours --worker examples/headline_ours_worker.py \
  --cell Qwen3_ShareGPT_R4_C30_B16_L512_O64 --repeats 1 --expert-executor native --native-prefill \
  --prefetch-off --prefill-optimized --prefill-layout-fast --decode-layout-fast --policy LA_CA_NEAR \
  --post-generation-diagnostic          # repeat with ..._B64_L512_O64; GPUs 0,1,4,5 via the guarded launcher
CUDA_VISIBLE_DEVICES="" python3 experiments/ferry_appendix_sensitivity_20261011/metadata/scripts/harvest_metadata_time.py OUT.csv OUT_SUMMARY.csv   # after adding the new run dir to RUNS
```

## 6. Parity / correctness (PARITY_CHECKS.json): overall PASS

- **A. New CPU harness: 646/646 PASS.** 4 gloo ranks; Qwen and DeepSeek geometries; B ∈ {1..256}; 3 decode steps × all layers. It checks:
  - record bytes match the formula;
  - header drift assert;
  - Top-K IDs survive in rank order;
  - Σ_{r,e} d_{r,e} = R·B·k, and each rank's row sums to B·k;
  - gate equals an independent `GateHistory.update` reference, is finite, and sums to 1 within 1e-4;
  - F = 𝓔 \ resident;
  - balanced quota with Σq = |F|;
  - identical controller state digest (fetches + slots + owner + primary) on all ranks every event;
  - rerun determinism;
  - the cap is lossless.
- **B. Existing CPU test:** `scripts/test_prefill_tail_metadata.py` PASS, 7 distributed cases.
- **C. Receipts.**
  - All 8 Qwen main cells: identical cross-rank `state_hash` with validation PASS in warmup and all 3 targets; prefill `metadata_exact_checks=48` on every rank.
  - **The DeepSeek main cells record no cross-rank state hash.** Their replicated-decision parity is not evidenced by receipts, only by the synthetic CPU check.
- **D. Real-trace conservation.** Across ranks, executed (token, expert) rows per event = R·B·k on every one of the 3,024 events for B8, B16 and B64 (Qwen diagnostics).
- **E. Earlier GPU preflight (2026-10-07).** `test_live_metadata.py` (40 checks per rank) and `test_async_metadata.py` (64 checks per rank) PASS. Qwen geometry only.

**Not checked:**
- "No future routing information" rests on code inspection only: `collect` sees only current-event tensors.
- LiveMetadata with DeepSeek geometry has not been run on GPU.
- Actual NCCL wire bytes were not measured.

## 7. Caveats for the appendix text

- Do not quote "24+B·k+4E" as FERRY's payload. The main cells used the live record.
- Do not call the all-gather spans transfer time. The flat ~110 µs floor across a 4× payload change, and the large rank-dependent spans, show latency and skew dominance.
- B_g/quota profiling is not part of the R4 main-cell decision path.
- The live record re-sends 2× (B64) or 4× (B≥128) more probability rows than the W=128 history can use.
