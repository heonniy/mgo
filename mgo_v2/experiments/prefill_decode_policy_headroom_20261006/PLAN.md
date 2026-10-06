# Prefill / Decode Placement Headroom Plan

Status: PLAN ONLY. This commit does **not** authorize or launch GPU execution.

Date: 2026-10-06  
Branch: `codex/policy-regime-20261005`

## 1. Goal

Measure whether rank placement matters more in **prefill / TTFT** than in **decode / TPOT**, while preserving the current R4 CPU-expert-offload setting.

The questions are:

1. For prefill, which of **BR / CA / LA** minimizes TTFT?
2. For decode, which of **BR / CA / LA** minimizes TPOT?
3. Does the best prefill policy remain best at decode, or are the two phases different operating regimes?
4. What is the maximum observable placement headroom when both the workload seed and placement/random seed are selected in favor of each candidate policy?

This is intentionally an **upper-bound / headroom characterization** first. Policy-specific seed selection is not a paper-final unbiased comparison.

## 2. Fixed hardware and runtime scope

Use only:

- physical GPUs: **0, 1, 4, 5**
- world size: R4
- GPUs 2, 3, 6, 7: **never use**
- model / precision: current Qwen3-30B-family MoE, BF16
- substitution: OFF
- replication: OFF
- decode horizon: 64
- no policy changes other than rank placement/admission
- no concurrent B5 mutation: keep the existing B5 result path separate

Primary matrix:

| Cache | local batch |
|---|---:|
| C30 | B16 |
| C30 | B128 |
| C60 | B16 |
| C60 | B128 |

Policies:

- BR
- CA
- LA

## 3. Phase isolation

Do not infer prefill and decode behavior from one all-phase policy run only.

### A. Prefill-headroom track

For each candidate `P in {CA, LA}`:

`P during prefill -> BR during decode`

Compare against:

`BR during prefill -> BR during decode`

Primary metric: **TTFT**.

Secondary metric: decode TPOT after the prefill boundary, to quantify the cache-state carryover caused by prefill placement.

This makes the prefill question causal: only the prefill policy changes.

### B. Decode-headroom track

Use a common prefill:

`BR during prefill -> P during decode`

Compare against:

`BR during prefill -> BR during decode`

Primary metric: **decode TPOT**.

The decode comparison must start from the identical BR-prefill boundary state. This avoids attributing a different prefill-created cache layout to the decode policy.

### C. Joint phase policy check

Only after A/B show nontrivial headroom, run the small phase matrix:

`prefill_policy x decode_policy`

for BR / CA / LA.

The joint matrix is for answering whether the final design should use one policy globally or a phase-specific policy pair.

## 4. Prefill communication barrier semantics

The current B5 post-expert barrier applies only to decode. Add an analogous **prefill diagnostic barrier**.

For every prefill MoE layer:

```
routing / admission
 -> async required expert H2D
 -> forward A2A dispatch completion
 -> rank-local required H2D + expert execution
 -> local current-layer compute-stream synchronize
 -> GLOBAL dist.barrier()
 -> return A2A
 -> combine
```

Important:

- wait only for work required by the **current prefill layer**;
- do not drain unrelated future/speculative transfers;
- record barrier time separately from return A2A;
- preserve the B5 decode barrier for the decode phase.

The purpose is the same as B5: keep slow-rank arrival skew from being mislabeled as pure NCCL return communication.

## 5. Metrics

### Prefill

Primary:

- TTFT wall time

Breakdown:

- prefill total wall time
- required H2D copies / bytes per rank
- max/min per-rank mandatory fetches
- per-rank expert token load
- forward A2A bytes / packet counts
- post-expert local-complete time
- post-expert global-barrier time
- return A2A time
- rank arrival / completion skew

### Decode

Primary:

- TPOT over decode64

Also report:

- E2E
- the same rank H2D/load/communication/barrier breakdowns
- boundary cache-state hashes before decode starts

For all comparisons, report both absolute time and percentage gain relative to the paired BR run.

## 6. Policy-specific seed search: upper-bound protocol

The requested seed search is allowed, but it must be labeled **BEST-SEED HEADROOM / ORACLE SCREEN**, not a general-policy result.

Search two independent seed axes:

- `workload_seed`: dataset sampling + rank assignment / request composition
- `placement_seed`: random initial placement / randomized tie behavior where the policy actually uses it

Initial search domain:

- workload seeds: 0..31
- placement seeds: 0..31

This gives 1,024 seed pairs per regime before pruning.

### Critical fairness rule

For a candidate policy, never compare its best seed to BR on a different workload.

For each candidate policy `P` and seed pair `s=(workload_seed, placement_seed)`, compute the gain using **BR on the exact same seed pair**:

[
G_{TTFT}(P,s)=\frac{TTFT_{BR}(s)-TTFT_P(s)}{TTFT_{BR}(s)}
]

and similarly

[
G_{TPOT}(P,s)=\frac{TPOT_{BR}(s)-TPOT_P(s)}{TPOT_{BR}(s)}.
]

Then select separately:

[
s^*_{prefill,P}=\arg\max_s G_{TTFT}(P,s)
]

and

[
s^*_{decode,P}=\arg\max_s G_{TPOT}(P,s).
]

Thus CA and LA may each use a different favorable seed pair, but every claimed gain is still a paired BR-vs-candidate comparison on identical requests and initial random state.

### Seed-activity audit

Before interpreting `placement_seed` as an optimization axis:

- hash the initial/cache placement and tie decisions;
- verify that changing the placement seed actually changes the evaluated policy state;
- if CA or LA is deterministic for a regime, mark `placement_seed_inactive=true` rather than inventing extra randomness.

## 7. Search and measurement stages

### S0. CPU / trace seed screen

Replay all 32 x 32 seed pairs without GPU timing.

For every regime and candidate policy, rank seeds separately for prefill and decode using controller/trace features:

- max-rank mandatory H2D work
- max-rank expert token load
- remote token->rank traffic
- rank imbalance / predicted critical-path proxy
- cache-state / future-miss effects

Keep the top **8** seed pairs per candidate and phase.

S0 is only pruning; no speed claim comes from the proxy.

### S1. Physical one-shot seed screen

On GPUs 0,1,4,5 only:

- one warmup / correctness pass;
- one clean timing sample for each retained seed;
- always run candidate and BR as a paired comparison on the same seed;
- counterbalance order.

Select the best physical seed per candidate and phase.

This timing is selection data, not the final estimate.

### S2. Final best-seed validation

For each selected seed:

- fresh clean process / common runtime;
- 2 repeats;
- third repeat if the current stability rule requires it;
- paired order reversal;
- report mean/median/range and instability flag.

For final decode measurements, use the current validated H1b substrate where feasible. Because H1b signatures are workload dependent, capture/validate signatures only for the selected final workloads rather than for the full seed grid.

## 8. Physical-work parity and interpretation

For every BR-vs-candidate pair record:

- total expert H2D copies
- total expert H2D bytes
- per-layer mandatory miss count
- per-rank H2D distribution
- final and prefill-boundary cache hashes

Classification:

1. **STRICT_PLACEMENT**: equal total physical expert copies/bytes and equal logical routed work; only rank distribution differs.
2. **PLACEMENT_STATE_EFFECT**: placement changes later eviction/residency and therefore total H2D work. Report separately as a downstream state effect.
3. **INVALID**: token/routing mismatch, substitution/replication drift, canceled speculative work mismatch that cannot be reconciled, or numerical/correctness failure.

Do not mix class 1 and class 2 into a single causal claim.

## 9. Decision gates

This experiment is intended to determine whether placement is worth continuing.

### Prefill

- **GO**: stable best-seed TTFT gain >= 5%
- **MARGINAL**: 2% <= gain < 5%
- **STOP_PREFILL**: even the policy-specific best-seed headroom is < 2%

### Decode

- **GO**: stable best-seed TPOT gain >= 3%
- **MARGINAL**: 1% <= gain < 3%
- **STOP_DECODE_PLACEMENT**: even best-seed headroom is < 1%

The final research direction can still be:

- prefill: meaningful TTFT placement optimization;
- decode: smaller phase-specific TPOT optimization;

if prefill provides the dominant gain and decode remains consistently positive.

## 10. Required result table

Produce one row per regime / phase / candidate:

| cache | batch | phase | candidate | best workload seed | best placement seed | BR metric | candidate metric | gain | copies parity | barrier share | return-A2A share | stability |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|

Also report the winning phase policy for each of:

- C30/B16
- C30/B128
- C60/B16
- C60/B128

The final summary should answer directly:

```
Best prefill policy by regime: ?
Best decode policy by regime: ?
Prefill maximum TTFT headroom: ? %
Decode maximum TPOT headroom: ? %
Does a phase-specific BR/CA/LA choice beat one global policy?
```

## 11. Safety / non-goals

- Do not use GPUs 2,3,6,7.
- Do not run GPU experiments as part of this PLAN commit.
- Do not change substitution or replication.
- Do not call best-seed results an unbiased average-case benchmark.
- Do not merge the prefill experiment with B5 output directories.
- Do not claim communication gain from packet reduction alone; use measured critical-path timing.
