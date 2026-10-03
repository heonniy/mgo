# Execution conventions and owner amendments

Owner message after 986ba64 explicitly authorizes stopping all GPU work and
using GPUs 0--7. Existing model load workers were stopped; all GPUs were empty.
The preceding dynamic refresh study remains STOPPED_BY_OWNER at 99/120 cells.

The owner then selected **both decode64 and decode256 for CPU replay**. This
extends the main matrix from 768 to **1536 cells**, not the number of model
captures. The extra BR seed audit remains at most 16 replays total. No further
axes, repeated captures or tuning are authorized.

## Exact master capture

Use eight resident, native BF16 Qwen replicas, one per physical GPU and logical
rank, each loaded once in the same session. Request i belongs to rank i%8.
Each rank handles 64 requests. Capture MATH, audit it, then capture ShareGPT
without reloading. These are independent exact-model trajectories with no
offloading/substitution or physical communication measurement. All logical
placement and communication comparisons are offline.

Native SDPA, deterministic greedy decoding, deterministic Torch kernels,
TF32 off, Qwen's user chat template. Store raw per-token router probabilities
as float32, selected expert IDs, and the actual BF16 execution weights converted
losslessly to float32. Padding tokens are excluded from prefill routing.
Store prompt IDs/masks, decode token IDs and ordering hashes outside git.

To honor both exactly 256 generated tokens and one prefill + 256 decode
forwards: prefill produces g1; decode forwards consume g1 through g256. Discard
the final logits (do not produce g257). EOS is suppressed during all 256 greedy
selections, matching min_new_tokens=max_new_tokens=256. Every decode request
is active. Decode64 is the first 64 consumed generated tokens/route steps of
this same trace. There is no second generation or warmup.

MATH selection balances subject/difficulty strata, seed42, then shuffles the
512 selected requests. Rendered inputs are right-truncated at 512 tokens.
ShareGPT V3 cleaned selection uses seed44, unique user prompts, 32--512 raw
prompt tokens and >=128 reference tokens; additionally require the rendered
prompt to fit 512, preserving the complete selected user turn. No reference
answer is appended. Source revisions, filters and manifests are hash-pinned.

## Resource bounds

Start only with all GPUs free (>=76 GiB each) and >=512 GiB host available.
GPU allocator cap is 88%; stop on <8 GiB GPU free or <256 GiB host available.
Stream traces to memory-mapped files rather than retaining Python records.
Monitor foreign GPU jobs and fail the capture if they appear. No automatic
retry, capture duplication, or idle worker restart during the session.

CPU policy replays use CUDA hidden, BLAS/OMP1, <=16 concurrent cells,
aggregate RSS <=32 GiB, and >=512 GiB available before launching a wave.
Completed cells must be hash-verified and reused, never silently rerun.

## Horizon audit definitions

Demand means raw selected expert route count. Quantiles include zero-demand
expert/rank pairs and explicitly distinguish event demand from horizon totals.
The top 10% means ceil(.1*128*8)=103 pairs per layer; stable flat-index ties.
Recurrence excludes the final step from future-opportunity denominators.
Cross64 recurrence asks whether a pair seen in steps 1--64 appears in 65--256.
Hot overlap compares the 103 highest-demand pairs of these two disjoint spans.

## Admission interpretation

CA follows the plan's explicit balanced assignment maximizing local effective
expert-route demand. This minimizes return expert rows; dispatch rows are
coalesced per token/destination, so the plan's parenthetical equivalence to
minimizing total exact peer bytes does not generally hold. Report recomputed
dispatch + return bytes, without claiming global optimality for their sum.
No future information is used in CA. All further tie and replica-score rules
will be fixed before CPU execution.
