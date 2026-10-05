# Owner amendment: R4 and decode64

Replaces the remaining R8/decode256 physical queue. Use physical GPUs 0,1,4,5, local B128 and B256, BF16, 64 decode steps. No new trace capture.

Use the last four logical request shards (old ranks4..7) from each frozen R8 input, renumbered0..3. Physical GPU IDs are separate from source shard IDs. Keeping the final source shard preserves the frozen rank-order Gate W128 tail for B>=128. Retain exact selected experts, routing weights, teacher tokens and first64 decode events. Build new R4 CPU cache/prefetch proofs. Total MAIN capacity remains3686 (60% of6144 experts), distributed922/922/921/921. Global batch becomes512/1024. These are new R4 observations and must not be pooled with R8.

Reuse the BR-only frozen P2/T2 as an explicit transferred configuration, not a claim of R4-optimal tuning. V1 P0 barrier, V2 P2/T2 barrier, V3 P2/T2 overlap, common code and frozen inputs. No repeat of the previous tuning sweep. Primary comparison remains BR vs LA; secondary CA follows on the selected stable arm.

Two complete counterbalanced pairs first. If both E2E and TPOT differences are<=2% for each policy, stop without further samples even if the gain CI is wide. Otherwise at most one third pair. Three-sample full spread must be<=5% and paired gain range<=2 percentage points to be eligible. Unstable cells are recorded and excluded; no five/seven repeats or reruns to obtain favorable measurements. All samples and uncertainty remain visible. Positive claims require supported paired intervals.

R8 was stopped at owner request during LA repeat4. Its completed three BR/LA pairs remain historical; interrupted work is not a timing sample. Old source/input fingerprints are not compared against this new R4 runtime.
