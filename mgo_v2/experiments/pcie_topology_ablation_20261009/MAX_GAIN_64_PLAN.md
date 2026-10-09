# Owner-requested best64 ShareGPT search

The owner asks to find 64 samples that give the maximum G-NEAR benefit over
R-NEAR. R means rank-order quota, not random; both use Near placement. BR is
the random placement policy. This supplemental search is separate from the
unchanged frozen headline64 workload and does not replace the original full
experiment goal.

Objective: maximize measured reduction of median TPOT for a frozen batch of 64
distinct real ShareGPT requests. Keep global64/local16, input512/output64, C30/P2,
greedy decoding, seed42, physical GPUs0,1,4,5, NUMA-shared 54-GiB pinned sources
per node, no prefetch and globally serialized communication/H2D/compute. Use
the same grouped-decode/native-metadata implementation for R and G. Preserve
all tested candidates and report the largest observed benefit, not a guaranteed
global optimum among all combinatorial batches and not corpus-average gain.

1. Finish the current optimization cohort and its diagnostics. CPU corpus
   scanning starts paused and cannot run during its primaries/diagnostics.
2. Census the ENTIRE frozen ShareGPT JSON, not only the historical 2,048-request
   prefix pool. One first eligible user-ending prefix per source conversation;
   use the original Qwen chat template and last512 real tokens. Never pad,
   duplicate prompts, alter model weights, or introduce synthetic inputs.
   Store raw tokenized pool and selected token manifests only under data2/esjung.
3. Form 32 preregistered candidate64 batches covering uniform random samples,
   coherent token-similarity groups and mixed groups from across the entire
   eligible pool. Require distinct source rows and distinct token
   prefixes within every batch; exclude the fixed warmup sources/prefixes.
   Include the original headline batch as a reference control. Dataset split
   records remain valid distinct requests; also report their original
   conversation-family multiplicities instead of claiming all rows are
   independent conversations.
4. Use a separately labelled short nomination capture (16 generated tokens)
   and independent-cache native C++ replay to rank predicted fetch opportunities
   on common captured routing. This is nomination data, not serving performance;
   count proxies do not establish physical speedup. Freeze the nomination method
   before observing candidate timings.
   Preregistered score: `1 - sum(G critical-group fetches) / sum(R critical-group
   fetches)` over the 15 captured decode forwards (exclude prefill). Replay both
   native C++ policies independently from empty caches on the captured G routes;
   G's full 61-column trace and final cache hash must match the actual capture.
   Break score ties by candidate label. Coincident lexical candidates use a
   deterministic uniform fallback seed `100000 + candidate_position`, selected
   before nomination; no timing-based replacement.
5. Physically screen the top eight nominations at full64 output, one R/G pair
   each with alternating pair order. Then freeze the top three by observed TPOT
   benefit and run at least three counterordered unprofiled pairs for each;
   predefine two more for all three if any arm TPOT spread exceeds 5%. Separate
   diagnostics verify physical phase order, pinned sharing, cache slots, native
   controller decisions, quotas, finite logits and no compilation in primaries.
6. Publish all screen/final repetitions, selection versus final timing, TPOT
   median/mean/sample SD/range, H2D and communication breakdown, M histogram,
   request IDs/provenance and manifest hashes. Label greedy R/G trajectory
   changes and selection bias. Deliver the winner's 64 request IDs and exact
   token manifest path, and commit each completed experiment separately.

If the best candidate has small or negative benefit, report that measured result
and continue diagnosis; do not redefine the winner as H2D-only gain or change
quotas/cache budget to manufacture an improvement.

Input-identity repair (before screen primaries): different source records can
have identical ordered token matrices. The initial source-index signature missed
two duplicate pairs, family0/1 and family3/4 (31 unique inputs among 33 labels).
The preregistered fallback now compares exact ordered uint32 input-token hashes,
so family1 and family4 use the predeclared seeds100025 and100028. All other31
manifest hashes are unchanged. Preserve the original nominations and interrupted
screen initialization, run only the two repaired nominees, verify identical
active runtime source hashes and warmup tokens/cache, then compose the31 valid
original nominees plus the two repaired nominees with per-case source provenance.
Recompute the frozen top8 from all33 unique candidates before any screen primary.
This repair does not use measured timing to construct candidates.
