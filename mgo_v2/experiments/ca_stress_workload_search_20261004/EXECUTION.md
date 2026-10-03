# Frozen implementation and execution handoff

Owner packet: `bcff29fc97572201f5b2a1131e514ee20c026322`.
Start after the timing-stability diagnostic reaches its scientific stop,
whether HARNESS_STABLE or HARNESS_UNSTABLE. An infrastructure failure requires
recovery of that diagnostic first. No physical policy timing follows.

## Candidate traces

Preserve both original 512-request manifests and exact trace shards unchanged.
For MATH, exclude their source rows, use seed42 stratified subject/difficulty
round-robin selection and shuffle for the additional requests. For ShareGPT,
reconstruct the original deduplication and token-length filter, verify its
eligible count and original prefix, then sample remaining eligible turns with
seed44. Do not relax either filter. Retain the same local checkpoint, prompt
template, right-truncation rule for MATH and exact generation settings.

Capture only request IDs >=512, at most 1536 additional requests per dataset.
All eight GPUs each load the native model once and sequentially capture three
waves of at most 64 requests/GPU for each dataset. Each request has one prefill
and 256 decode forwards; generated tokens and full router probabilities are
hash-pinned. This is route capture, not timing. Large artifacts remain under
`/home/hwlee/mgo-results/ca_stress_workload_search_20261004/`.

## Stage A proxy, frozen before search

Evaluate all sample_seed 0..255 by dp_seed 0..255 for each dataset/R/B group.
Sampling uses NumPy default_rng(sample_seed).permutation(pool_size)[:R*B];
default_rng(dp_seed) permutes that subset and consecutive B requests form each
rank. Inspect exactly 48 raw-route events: each layer 0..47 once, using decode
array index `(layer % 4) * 85` (0, 85, 170, 255). This deterministic sample spans
all layers and four horizon positions without a cache replay.

For each event, count exact top8 route demand per expert and rank. Treat its
active experts as instantaneous mandatory admissions with balanced floor/ceil
rank quotas. In descending best-minus-second-best demand order (stable expert
ID ties), greedily assign an expert to its highest-demand rank with a remaining
slot. Subtract expected local-route count of random assignment to the same
slots. Average this difference over the 48 events. This feasible greedy score
is a cheap locality proxy, not the Hungarian optimum or cache-policy savings.
Retain the 32 distinct highest-scoring sample/DP pairs; ties use ascending
sample then DP seed. No BR seed is selected by this proxy.

## Exact replay and reporting

Stage B uses the unchanged `br_carep_cpu.py` replay: Gate W128, substitution OFF,
256 decode steps, cache30/40/50/60. Per candidate/cache run deterministic CA
once and BR seeds [7,19,42,73,99,131,181,251]. The existing BR shuffle and CA
Hungarian solver share the same per-current-event floor/ceil admission slots;
assert max-minus-min rank admissions <=1. Future miss sets may differ.

Before the search, reconstruct original MATH/R4/B8 inputs from the normalized
pool and require exact array equality with the archived pack. At cache30,
require BR42 and CA final-state hashes and principal resource counters to
reproduce archived full256 results. This is a compatibility gate, not a new
scientific axis. Trace files are hash-audited before normalization.

There are 1,048,576 Stage A pairs, 512 retained candidates, 18,432 exact
policy replays, 64 winning cells and 192 top-three rows. Primary selection is
median absolute peer-byte reduction over the eight BR seeds, then median
relative reduction, then lower CA peer bytes. Residual ties use ascending
sample/DP seeds. Primary resource totals include prefill and all 256 decode
steps; decode-only counters are also preserved in candidate receipts.

Report best-observed BR seed separately, full gain ranges, H2D and hit rates,
balanced request manifests, route/Gate and final-state hashes. Reuse the
archived unoptimized seed42 workloads as neutral controls, explicitly noting
that a single-seed control is not the eight-seed median. Selected rows are
optimized communication-stress examples, not dataset-average performance.

## Resource and completion rules

No heavy preparation, JIT compilation or CPU search runs alongside diagnostic
timing. Additional GPU capture enforces host/GPU memory, temperature and
foreign-process guards. After GPU science ends, restore the owner's eight
`model_inference_load` workers while CPU-only processing continues.

CPU workers use read-only shared file mappings and single-thread BLAS/Numba,
with disjoint CPU affinity, at least 16 host vCPUs reserved on this 192-vCPU
host, 8-GiB private-data and 96-GiB address-space limits per worker. Start with
eight useful jobs, grow concurrency while normalized throughput improves by
at least 5%, bounded by CPU affinity and an 8-GiB/worker budget after reserving
320 GiB. Abort and preserve checkpoints if host available memory drops below
256 GiB. Record wave throughput and summed RSS (which double-counts shared
pages); never interpret that sum as physical memory consumption.

Commit/push manifest, capture, normalization, replay and final-result phases.
On failure preserve the stage/error and all completed receipts. The outer
chain restores/verifies all eight resident model workers on completion or
failure. The older physical policy matrix and dynamic-refresh study stay
paused. No further experiment follows automatically.
