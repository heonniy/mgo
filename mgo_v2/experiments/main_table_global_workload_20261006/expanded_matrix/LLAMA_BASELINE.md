# Selected llama.cpp baseline implementation

Owner-approved following balanced3 validation at e847735. Static12 GPU expert
layers, exactly3 on each physical GPU0/1/4/5. CPU holds remaining36 layers.
Each GPU expert residency3.375GiB (384 experts), global13.5GiB, below C30 cap.
Unused C30 space is intentional whole-layer granularity; do not call it fully
utilized30% caching. CPU32/32 with deterministic affinity; CUDA Graph and
ordinary graph reuse OFF. BF16; synchronous batch; attention/KV on GPU;
op_offload=false. Output64, common TTFT/TPOT=(E2E-TTFT)/63/E2E definitions.

Use the dedicated launcher, which fixes baseline flags and rejects canceled
C60/B64 scope:

```bash
python mgo_v2/scripts/run_llama_baseline.py \
  --cell R4_C30_B32_L512_O64 --label NEW_UNIQUE_LABEL --repeats 2
```

Use `--dry-run` to inspect without GPU work, `--smoke` for four requests,
32 input tokens and two output tokens. Supported full cells: C30 B16/B32 x
input256/512. A new run label is required; old data cannot be overwritten.
Warmup is disjoint, KV is cleared before each primary, static expert weights
remain. Supervisor retains host384/96GiB guards and foreign-process protection.

The Python worker defaults to balanced3. Historical thread/CUDA/reuse drivers
explicitly request legacy_tail to preserve their experiment identity. The
native low-level runner keeps its backwards-compatible legacy argument default;
the baseline always passes balanced3 explicitly. New queue plans include the
placement flag and distinct balanced3 labels; old completed QUEUE.json and
RESULTS are preserved, and canceled cells are excluded from plan generation.

Actual loader ownership must be CUDA0/1/2/3 with counts3/3/3/3 and exact CPU
expert exclusions. Fail before timing if ownership differs. No guessed device
mapping, suppressed validation, CPU attention, or extra replica mechanism.

Validation: existing full B32/L512 two primaries passed; new launcher CPU checks
cover all4 supported cells, rejected C60 and corrected queue scope. A new small
GPU smoke validates the selected launch path. This is implementation adoption,
not a new main-table result. Legacy versus balanced3 generated outputs differ
(1720/8192 token positions in previous comparison); no output-equivalence claim.
Old14-layer timing is retained separately. Do not automatically run a matrix.
