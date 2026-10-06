# Completion audit

Scope is the full requested R4 packet from7851e2ff and owner amendments, not a reduced subset. Execution is complete with explicit timing limitations. The original stable-panel gate remains failed for five rows; no report or verifier promotes them to PASS.

| Requirement | Current evidence | Outcome |
| --- | --- | --- |
| Both matrix cells, all four systems, OURS Near | HEADLINE_R4_MATRIX.json; exact eight-row coverage and system/cell checks in EXECUTION_AUDIT.json |8/8 executed and validated |
| Same frozen global64/256 request manifests; disjoint warmup; input256/512;64 outputs | Manifest SHA/order checks, output shapes and raw timestamps in both audit scripts and all archived primary receipts | PASS |
| R4 physical0/1/4/5 only; shared-server memory protection | Worker/supervisor launch receipts, resource logs,384/96GiB host guards; FINAL_HANDOFF.json | PASS; no other GPUs modified |
| BF16 model, native greedy/no frozen routing or teacher tokens | Worker code and versioned configs; GGUF_METADATA.json; MoE BF16 guard | PASS with documented small F32 GGUF tensors and native FP16 llama KV |
| GPU attention/KV; no CPU KV offload | OURS CUDA dense/attention construction; DeepSpeed per-batch KV checks; MoE attention/KV guards; llama smoke allocation and unchanged all-layer GPU assignment | PASS at documented evidence scope; no full llama per-layer primary telemetry claim |
| Warmup then cold dynamic expert state before each primary; prefill cache carries into decode | Runtime reset/cold receipts for all dynamic systems; llama fixed partitions and disabled prompt reuse | PASS |
| Exact C30 expert limit and declared host cost | OURS1843 physical slots including prefetch; MoE per-GPU native cap/peak counters; conservative DS all-parameter cap; llama14layers/15.75GiB | PASS; total HBM/RSS/pinned costs reported separately |
| MoE priority eviction and EAM actually exercised | Every final primary has positive EAM/candidate/priority counters;9 Python and15 native tests; physical dispatch/chunk/KV regressions | PASS; explicitly labelled repaired baseline |
| Three unprofiled samples; global TTFT/TPOT/E2E/throughput; preserve all samples | Raw-clock recomputation and exact sample counts; FINAL_RESULTS.md and per-attempt audits | PASS |
| Bounded response to unstable timings | MAIN_TABLE_OUTCOMES.json records each earlier unstable triplet and one completed confirmation; all samples preserved | Completed;3 stable rows,5 unstable rows. No stability-qualified full-panel winner |
| Provenance and actual patch preservation | BASELINE_VERSIONS.json; launch commit/hash receipts; PROVENANCE_CHECKPOINT.json; versioned patches | PASS |
| Results retained byte-for-byte and reviewable | ARCHIVE_CHECKPOINT.json verifies all files of all8 final reporting attempts; earlier attempts remain archived | PASS |
| Complete report and limitations | FINAL_RESULTS.md includes all8 timing ranges, throughput, resources, implementation differences, archives and unresolved variability | PASS |
| Workers/queues stopped; authorized idle model load restored | FINAL_HANDOFF.json plus live process/inventory validation | PASS on0/1/4/5 only |

Git ancestry from7851e2ff was checked. Final staging/commit/push and remote-HEAD equality are verified after this audit is written. Stable-panel eligibility is intentionally separate from completion of the requested bounded experimental work; no unfinished run or missing baseline is hidden by that distinction.
