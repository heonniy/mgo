# R4 main-table experiment results

> Correction: llama.cpp TPOT below is an asynchronous **batch-tail interval**, not mean per-request TPOT. It must not support direct decode-speed rankings against synchronized runtimes. See [timestamp audit](LLAMA_TPOT_SEMANTICS.md). Original measurements are preserved.

All eight matrix rows and their scheduled bounded confirmations completed. All final attempts pass execution/correctness checks. **Three rows satisfy the timing spread gate; five remain unstable.** This completes the requested experimental execution, but does not establish a fully stable eight-row headline comparison. No slow sample was removed and no open-ended repeat loop was added.

## Workload and reporting

- Qwen3-30B-A3B-Instruct-2507, GPUs0/1/4/5, R4, C30 expert budget1843 slots =17,392,730,112bytes (16.1982421875GiB global).
- Local B16/input256 means global64 requests; local B64/input512 means global256. All systems use identical frozen ShareGPT-long manifests. Native greedy generation ignores EOS and produces exactly64 tokens; no frozen routing or teacher forcing.
- One disjoint warmup, then three unprofiled primaries. Reset dynamic expert residency before every measured batch; retain compiled code/CPU backing/static partitions. Measured prefill residency continues into decode.
- TTFT and E2E use synchronized common release to global maximum first/last-token readiness. TPOT=(E2E−TTFT)/63. Each metric below is the independent three-sample median, so displayed medians need not satisfy that identity together.
- An unstable initial valid triplet receives only the already scheduled bounded confirmation. The explicit [outcome map](MAIN_TABLE_OUTCOMES.json) fixes which full triplet is reported. Earlier attempts remain in [the all-attempt audit](MAIN_TABLE_PROGRESS_AUDIT.json).

## Final reporting triplets

Seconds; entries show **median [minimum, maximum]**. UNSTABLE means any TTFT/TPOT/E2E spread `(maximum−minimum)/mean` exceeds5%. These rows must not support stability-qualified speedup claims.

| Cell (local B / input L) | System | TTFT seconds | TPOT seconds | E2E seconds | Timing |
| --- | --- | --- | --- | --- | --- |
| B16 / L256 | OURS Near | 4.041779 [3.992524, 5.816702] | 0.769798 [0.758135, 0.787128] | 52.539039 [51.755061, 55.405744] | UNSTABLE |
| B16 / L256 | MoE-Infinity repaired | 5.020634 [4.839624, 5.136096] | 3.073081 [3.067092, 3.112734] | 198.740211 [198.066392, 201.122880] | UNSTABLE |
| B16 / L256 | DeepSpeed CPU offload | 5.412409 [5.320577, 6.045086] | 4.085473 [4.065967, 4.095595] | 262.797200 [262.201012, 263.343088] | UNSTABLE |
| B16 / L256 | llama.cpp static | 37.702121 [37.441671, 38.006756] | 0.257713 [0.257713, 0.261402] | 54.170472 [53.677566, 54.242655] | PASS |
| B64 / L512 | OURS Near | 26.584922 [26.405346, 28.784070] | 1.256196 [1.119625, 1.288150] | 107.558812 [97.121306, 107.924442] | UNSTABLE |
| B64 / L512 | MoE-Infinity repaired | 9.650225 [9.523655, 9.674714] | 3.696202 [3.637762, 3.761964] | 242.535426 [238.829239, 246.527360] | PASS |
| B64 / L512 | DeepSpeed CPU offload | 7.927775 [5.692961, 8.420578] | 4.825910 [4.696274, 6.305956] | 311.960134 [304.285824, 402.968185] | UNSTABLE |
| B64 / L512 | llama.cpp static | 334.628984 [331.908713, 343.538012] | 0.404266 [0.403794, 0.415453] | 360.068001 [358.082243, 369.006798] | PASS |

## Output throughput

Global generated tokens per second; median [minimum, maximum] over the same triplet. All64 output tokens per request count toward throughput; the timing-status limits above still apply.

| Cell | System | Tokens/s |
| --- | --- | ---: |
| B16 / L256 | OURS Near | 77.961 [73.927, 79.142] |
| B16 / L256 | MoE-Infinity repaired | 20.610 [20.366, 20.680] |
| B16 / L256 | DeepSpeed CPU offload | 15.586 [15.554, 15.622] |
| B16 / L256 | llama.cpp static | 75.613 [75.513, 76.307] |
| B64 / L512 | OURS Near | 152.326 [151.810, 168.696] |
| B64 / L512 | MoE-Infinity repaired | 67.553 [66.459, 68.601] |
| B64 / L512 | DeepSpeed CPU offload | 52.520 [40.658, 53.844] |
| B64 / L512 | llama.cpp static | 45.503 [44.400, 45.755] |

## Resources

GiB=2^30bytes. HBM is the largest per-GPU1Hz NVML observation during measured repeats; short allocation peaks may be missed. Host RSS is end-of-repeat RSS, summed across ranks for OURS/DeepSpeed and possibly double-counting shared pages. It is not unique host physical RAM. Unknown pinned memory is unavailable, not zero.

| Cell | System | Max HBM/GPU GiB | Host RSS GiB | Observed pinned GiB |
| --- | --- | ---: | ---: | ---: |
| B16 / L256 | OURS Near | 14.22 | 606.39 | 216.00 |
| B16 / L256 | MoE-Infinity repaired | 11.47 | 59.67 | unavailable |
| B16 / L256 | DeepSpeed CPU offload | 11.86 | 90.70 | 56.87 |
| B16 / L256 | llama.cpp static | 14.84 | 41.28 | unavailable |
| B64 / L512 | OURS Near | 19.23 | 697.23 | 216.00 |
| B64 / L512 | MoE-Infinity repaired | 59.95 | 59.82 | unavailable |
| B64 / L512 | DeepSpeed CPU offload | 16.56 | 90.72 | 56.87 |
| B64 / L512 | llama.cpp static | 18.37 | 41.95 | unavailable |

Total HBM includes dense weights, KV, activations and workspace; C30 is the expert-residency budget, not a30GiB total allocation limit. OURS uses216GiB of rank-private pinned CPU expert backing. These weights remain in CPU memory until fetched into bounded GPU slots. Equal host footprint is not claimed.

## Implementation and comparison limits

- OURS is LA_CA_NEAR with H0/full-pinned, V3P2/T2 and1843 total physical expert slots including prefetch slots. All measured repeats pass cold-state, consistent-cache and no-compilation guards.
- MoE-Infinity is explicitly **repaired**: request-level EAM from completed observed traces; identical warmup history restored for each primary; priority/LRU eviction with lease protection; exact per-GPU byte accounting including transfers; safe event ownership, large-expert chunks and post-batch KV release. All final primaries exercise EAM and priority eviction within the byte caps.
- llama.cpp uses layer split with14 GPU expert layers (15.75GiB) and34 CPU expert layers. `--no-op-offload` keeps host expert operations on CPU. Prompt reuse is disabled. BF16 weights coexist with converter-native small F32 tensors and native **FP16 KV**. GPU KV buffers are physically verified at smoke shape; full-primary layer-by-layer allocation telemetry is not claimed.
- DeepSpeed is stock ZeRO3 CPU parameter offload with pinned host parameters, BF16 and GPU KV. It is not the MoE-Infinity-paper FastGen baseline. Its all-parameter residency bound is conservative relative to an expert-only bound.
- Framework/Transformers versions and runtime structures differ; this is a whole-system benchmark, not isolated policy attribution. [Version details](BASELINE_VERSIONS.json) and [source/patch continuity](PROVENANCE_CHECKPOINT.json) are preserved.

## Timing limitations

Both OURS cells and both DeepSpeed cells remain unstable after their bounded confirmations. MoE-Infinity small remains TTFT-unstable (5.93% spread), though its TPOT/E2E spreads are below2%. Only both llama.cpp cells and MoE-Infinity large qualify for the strict whole-row timing gate.

OURS small reproduces a longer first-primary TTFT in two launches despite cold expert-state and no-compilation guards. DeepSpeed large confirmation repeat2 has50of63 token intervals over5.5s on every rank; this is not one isolated spike. These observations do not identify a causal bottleneck. No claim is made that timing variability has been fixed, and no stable end-to-end winner is declared from the unstable rows.

## Reproducibility and closure

- [Execution audit](EXECUTION_AUDIT.json):8/8 requested outcomes complete,3/8 timing-stable. [Audit semantics](EXECUTION_AUDIT.md) distinguish execution completion from timing eligibility.
- [Stable selection](HEADLINE_SELECTION.json) contains only the three PASS rows. The older stable-panel audit correctly remains incomplete because five rows are unstable.
- [Archive integrity](ARCHIVE_CHECKPOINT.json): every final attempt receipt/log matches its raw original. All3 samples and earlier attempts are preserved.
- [Final handoff](FINAL_HANDOFF.json): all queues finished and experiment workers exited. Owned model-forward loads restored only on GPUs0/1/4/5. GPUs2/3/6/7 were not touched by this experiment.

Final attempt archives:

- `ours_B16_L256_confirmation2`: [confirmation_B16_L256_attempt2](ours_first_native_attempts/confirmation_B16_L256_attempt2/RESULTS.md).
- `infinity_B16_L256_confirmation1`: [confirmation_B16_L256_attempt1](moe_infinity_repair/confirmation_B16_L256_attempt1/RESULTS.md).
- `deepspeed_B16_L256_affinity2`: [confirmation_B16_L256_affinity2](deepspeed_validation/confirmation_B16_L256_affinity2/RESULTS.md).
- `llama_B16_L256_confirmation1`: [confirmation_B16_L256_attempt1](llama_measurement/confirmation_B16_L256_attempt1/RESULTS.md).
- `ours_B64_L512_confirmation1`: [confirmation_B64_L512_attempt1](ours_first_native_attempts/confirmation_B64_L512_attempt1/RESULTS.md).
- `infinity_B64_L512_primary3`: [primary_B64_L512_attempt3](moe_infinity_repair/primary_B64_L512_attempt3/RESULTS.md).
- `deepspeed_B64_L512_confirmation2`: [confirmation_B64_L512_attempt2](deepspeed_validation/confirmation_B64_L512_attempt2/RESULTS.md).
- `llama_B64_L512_static1`: [primary_B64_L512_static1](llama_measurement/primary_B64_L512_static1/RESULTS.md).
