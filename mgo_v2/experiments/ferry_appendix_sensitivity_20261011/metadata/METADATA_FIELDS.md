# FERRY metadata: what each stage needs (D1 / E5 field table)

Code ref: work tree `claude/ferry-appendix-sensitivity-20261011` at `a0d77a9` (paper_result). All paths are relative to `mgo_v2/`.
Main-paper OURS cells use `examples/headline_ours_worker.py` (Qwen) and `examples/headline_ours_deepseek_worker.py` (DeepSeek). Both run the **live** decode metadata path and placement policy 7 (`LA_CA_NEAR`, "Near"). See RESULTS.md §3 for the evidence.

**Implementation note for the appendix.** `GlobalExpertController` (`mgo_v2/controller.py:23-28`) carries the "deterministic, replicated on every rank" design. The main-table runtime does not instantiate it, though. The cells run the same replicated design through the array-backed `DecodePrefetchController` (`mgo_v2/controller.py:233-279`, Qwen), which wraps the numba `Policy` in `scripts/env_offload_policy.py:174-205`. The DeepSeek worker uses that `Policy` directly (`examples/headline_ours_deepseek_worker.py:112-113,171`). Every rank feeds identical all-gathered metadata into its own `Policy.apply`. The rows below cite those code paths.

Legend for **Class**:
- **local**: lives only on the producing rank and is not exchanged.
- **exchanged**: part of the per-layer-event all-gather payload.
- **derived (replicated)**: every rank recomputes the same value from exchanged data plus replicated state. It is never sent.
- **offline**: a constant or table fixed before the run.

R = ranks (4), B = local batch per rank, k = Top-K, E = routed experts per layer, L = MoE layers (Qwen 48; DeepSeek 26 = 27 layers − 1 dense, `first_k_dense_replace=1`), W = 128.

## A. Decode (one MoE layer event = one decode token step × one MoE layer)

| # | Item (paper symbol) | Class | Owner | dtype / shape (per event) | Update scope | Code evidence |
|---|---|---|---|---|---|---|
| 1 | Event header `[version=1, event, B]` | exchanged | each rank sends, all check | int64 ×3 = 24 B per rank | per layer event | `live_metadata.py:24` (pack), `:29-30` (all ranks' headers asserted equal to `[1,event,B]`) |
| 2 | Top-K expert IDs per token | exchanged | source rank (router output) | **uint8** [B, k] per rank (B·k bytes); gathered [R·B, k] in rank order | per layer event | pack `live_metadata.py:25`; unpack `:31`; uint8 requires E ≤ 256 (`:13`) |
| 3 | Router Top-K weights (combine weights) | **local** | source rank | BF16 [B, k] (stays on GPU) | per layer event | not packed. Controller gets a constant `ones` array (`live_metadata.py:21,36`). The real weights stay on the token's own rank for combine (`headline_ours_deepseek_worker.py:191-193`; Qwen `decode_runtime.py` dense weights) |
| 4 | Full pre-TopK router probabilities (gate input for H) | exchanged (**live** path: only the last min(B,128) rows per rank) | source rank | float32 [min(B,128), E] per rank | per layer event | `live_metadata.py:16` (`tail=min(batch,128)`), `:26` (pack), `:32` (unpack) |
| 4' | Gate summary (compact path only) | exchanged (compact) | each rank; only the **last rank's** vector is consumed | float32 [E] per rank | per layer event | `compact_metadata.py:12,38-46,53`; B<128 requires a frozen trace gate (`:26-27`). Not used by main cells |
| 5 | Source-rank × expert demand d_{r,e} | derived (replicated) | every rank | int64 [R, E] histogram; the policy recomputes its own int64 [E, R] `demand` | per layer event | `live_metadata.py:35` (bincount over gathered IDs and origins); `env_offload_policy.py:42,61` (policy's own `demand[target,r]`). Conservation Σ d = R·B·k: PARITY_CHECKS.json |
| 6 | Origin rank of each gathered token | derived (static) | every rank | int64 [R·B] = repeat(arange(R), B) | per run (fixed B) | `live_metadata.py:20` |
| 7 | Required expert set 𝓔 = {e : d_e > 0} | derived (replicated) | every rank | bool [E] `active` | per layer event | `env_offload_policy.py:42,61` |
| 8 | Residency C_r (cache membership) | derived (replicated) state | every rank holds an identical copy | `slots` int32 [R, max C_r] (global key = layer·E + e, −1 = empty); `owner` int16 bitmask [L·E]; `primary` int8 [L·E] | mutated per layer event by fetch/evict | `env_offload_policy.py:178` (state), `br_carep_cpu.py:66-87` (`place` updates on admit/evict); cross-rank equality asserted every repeat (`headline_ours_worker.py:40-46`) |
| 9 | Resident-for-this-layer view | derived (replicated) | every rank | bool [E] | per layer event | `env_offload_policy.py:25` |
| 10 | Free / evictable slots u_r | derived on demand (**not stored**) | every rank | implicit: first empty slot, else min-gate non-active victim | per admitted miss | `br_carep_cpu.py:53-63` (`choose_slot`: empty slot first; skips same-layer *active* experts at `:58`) |
| 11 | Active-expert protection | derived (replicated) | every rank | bool [E] `active` (same as #7) | per layer event | `br_carep_cpu.py:58`. The `maxweight≥.20` protection (`env_offload_policy.py:26`) only matters for substitution, which is OFF |
| 12 | Resident compute load N_r (expert-token rows already owned) | derived (replicated) | every rank | int64 [R] `loads` | per layer event (recomputed, then incremented per placed miss) | `la_placement.py:27-52` (`estimate_loads`), `:124`, `:164` |
| 13 | Gate history H^{(ℓ)} (W=128 rolling window of probabilities) | derived (replicated) state | every rank holds an identical copy | per layer: deque of ≤128 float64 [E] rows plus `sums` float64 [L, E]. Gate score = float32(sums/len) [E] | per layer event (decode); prefill seeds it | `eviction.py:26-34`, `:56-78` (`update_compiled`), `live_metadata.py:33-34`; prefill `decode_runtime.py:61,69-71` / `headline_ours_deepseek_worker.py:162-166` |
| 14 | Gate scores used by eviction | derived (replicated) | every rank | float32 [L, E] `gates` (row ℓ overwritten each event) | per layer event | `env_offload_policy.py:19`; used in `br_carep_cpu.py:59` |
| 15 | Missing set F = 𝓔 \ ∪_r C_r | derived (replicated) | every rank | int64 [|F|] | per layer event | `env_offload_policy.py:64` |
| 16 | Quotas q_r | derived (replicated) | every rank | int64 [R]. Main cells use the balanced split q_r = ⌊|F|/R⌋ + [r < |F| mod R] | per layer event | `la_placement.py:117-122`; checked in `controller.py:268-269` and `env_offload_policy.py:129`. Offline quota LUT only for R8 policies NEAR_PCIE/FAST/SPLIT (`controller.py:236-246`) |
| 17 | Bandwidth groups B_g / per-rank copy cost | **offline**; **not used by R4 main cells** | profiler (offline) | JSON quota LUT int64 [129, 8] (R8 only); `H2D_US_BY_RANK_R8` table (HAQ_FAST only) | per run (static) | `controller.py:236-246`, `env_offload_policy.py:11,70-73,107-109`. Policy 7 ignores them (`env_offload_policy.py:110-112`) |
| 18 | Expert size S | offline constant | config | Qwen 3·2048·768·2 = 9,437,184 B; DeepSeek 3·2048·1408·2 = 17,301,504 B | static | `br_carep_cpu.py:6` (`EXPERT_BYTES`), `headline_ours_deepseek_worker.py:42` (`EXPERT_ELEMENTS=8650752`); model `config.json` |
| 19 | Assignment F → ranks (Near) | derived (replicated) | every rank | int64 [|F|] | per layer event | `env_offload_policy.py:110-112` → `la_placement.py:102-166` (heavy-first, `near_bps=200` ⇒ slack max(1, 2%·best)) |
| 20 | Evictions / admissions (fetch list) | derived (replicated) | every rank computes all; each rank issues only its own H2D | list of (rank, key, slot, victim, 0) | per layer event | `env_offload_policy.py:130-133`; own-rank H2D `decode_runtime.py:162-174`, `headline_ours_deepseek_worker.py:185-190` |
| 21 | Dispatch destinations per (token, expert) | derived (replicated) | every rank | int8 [R·B, k] | per layer event | `env_offload_policy.py:167-171` |
| 22 | LRU ticks / stats (`last`, `birth`, `reuses`, `seen`) | derived (replicated) state | every rank | int32 [R, L·E] etc. | per layer event | `env_offload_policy.py:179-180`; `last` is the tie-break in `choose_slot` |

## B. Prefill (first forward, one event per MoE layer)

The prefill path does not use LiveMetadata. It uses `gather_global_routes` (`mgo_v2/runtime.py:40-98`) with **padded** `dist.all_gather` calls:

| Item | Class | dtype / shape per rank | Code |
|---|---|---|---|
| token count | exchanged | int64 [1] | `runtime.py:57-64` |
| Top-K IDs | exchanged | **int64** [T_r, k] padded to max_r T_r | `runtime.py:66` |
| Top-K weights | exchanged | BF16 [T_r, k] padded | `runtime.py:67` |
| probability tail | exchanged | float32 [tail_r, E] padded to max_r tail_r. Only the global last 128 rows survive (`probability_tail=128`) | `runtime.py:68-80`; enabled by `decode_runtime.py:61` (Qwen, `--prefill-optimized`) and `headline_ours_deepseek_worker.py:162-163` |

The prefill byte formula is checked against the diagnostic receipt (`metadata_padded_send_bytes`, Qwen B16/L512: ids 25,165,824 + weights 6,291,456 + probs 3,145,728 B over 48 layers). This matches METADATA_BYTES_PREFILL.csv to the byte, apart from the 8 B/layer count all-gather.

## C. What is *not* sent on the per-event all-gather

Residency C_r, u_r, N_r, H, q_r, F, the assignment and the evictions are never transmitted. Each rank recomputes them from the exchanged IDs and probability rows plus its replicated copy of the state. The only per-event collective payload is items 1, 2 and 4 (live) or 1, 2 and 4' (compact). There is one decode all-gather per MoE layer: `dist.all_gather_into_tensor` (`live_metadata.py:27`). Cross-rank consistency of the replicated state is not exchanged per event. It is verified once per generation by an all-gathered digest (`headline_ours_worker.py:40-46`, Qwen only).
