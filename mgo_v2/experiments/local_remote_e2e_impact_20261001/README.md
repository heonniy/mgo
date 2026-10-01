# Physical local/remote TPOT and E2E study — 2026-10-01

This experiment measures two things in the validated mgo_v2 runtime:

1. how actual MoE/NCCL latency changes as expert service becomes more local or more remote;
2. how much the final communication-aware admission policy changes TPOT and end-to-end generation latency under real CPU expert offloading.

The runtime is already validated for R1/R4/R8 native parity, physical expert H2D accounting, direct resident-slot execution, substitution, Coverage eviction and all admission policies. This folder preserves the prospective contract and completed study results.

Do not retune policy coefficients after timing results are visible.

## Completed execution — 2026-10-02

See [RESULTS.md](RESULTS.md) for all five repeats, comparisons, figures, profile results and scope limits.

- R8 local batches 8, 4, 16 and 32 ran before R4 local batch 8: 15 conditions, 75 generations and 540 rank receipts.
- Both R4 stages used physical GPUs **0,1,4,5**, independently checked against worker processes and GPU UUIDs in [device observations](physical_device_observations.json).
- Stage A completed 30 event/map cells, with 600 uninstrumented and 600 separate diagnostic iterations. All outputs match bitwise and timed expert fetches are zero.
- [Primary validation](validation.json) and [two posthoc profiles](profile_summary.json) pass. Actual expert H2D matches logical fetch bytes on all 16 profiled rank/cell pairs; no full-expert D2D remains.
- [Performance acceptance](acceptance.json) is **NEGATIVE_RESULT**: C2 improves R8/B8 median TPOT by 5.01%, but worsens R4/B8 by 16.79%. The requested R8/B16 and B32 expansions worsen median TPOT by 2.95% and 20.36%, respectively. These are descriptive medians, with all ranges retained.

[Execution binding](EXECUTION_BINDING.md), [manifest](measurement_manifest.json), [execution receipts](execution_receipts.json), [trace hashes](raw_trace_receipts.json) and [artifact hashes](artifact_hashes.json) preserve provenance. Large raw traces and full token receipts remain at `/home/hwlee/mgo-results/local_remote_e2e_impact_20261001`.

For a fresh server reproduction, use `scripts/run_local_remote_study.py` from the `mgo_v2` directory with the bound model/store/inputs and a new output directory. Defaults include R8/B16/B32 before R4 and `--r4-gpus 0 1 4 5`; `--dry-run` writes the manifests and GPU/job order without launching CUDA. The summarize, posthoc profile, plot and report commands used for this execution are preserved in `execution_receipts.json`.

The optional 128-step confirmation was not run. The study is complete and stopped for owner review.
