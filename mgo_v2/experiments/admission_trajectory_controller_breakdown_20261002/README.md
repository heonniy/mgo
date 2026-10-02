# Admission trajectory and controller breakdown — 2026-10-02

The bounded packet is complete: **six R4 diagnostics on physical GPUs 0,1,4,5** and **60 single-process CPU replays**, with all correctness gates passing. No new R8 job was launched.

Start with [RESULTS.md](RESULTS.md). Detailed evidence:

- [Implementation and measurement definitions](IMPLEMENTATION.md)
- [Validation](validation.json) and [measurement manifest](measurement_manifest.json)
- [Controller breakdown](controller_breakdown.csv) and [implementation audit](implementation_audit.md)
- [Cache trajectories](trajectory_summary.csv), [next-use survival](next_use_survival.csv) and [state examples](state_examples.md)
- [Matched-demand comparisons](matched_comparisons.csv), [all replay repetitions](replay_repeats.csv) and [policy/stream decomposition](counterfactual_decomposition.csv)
- [Artifact hashes](artifact_hashes.json) and [raw-file receipts](raw_hash_receipts.json)

The original prospective [plan](PLAN.md), [matrix](matrix.json) and [agent task](AGENT_TASK.md) remain unchanged. The plan commit is `30614bbc065c140060ef31af9bc4f8615ab16ca5`; prior performance evidence is fixed at `e61758e`. The new runs are diagnostics, not new speedup estimates. Normal policy/runtime implementation and the native executor remain unchanged; instrumentation is opt-in.

Raw router streams, rank-event records and cache snapshots remain at `/home/hwlee/mgo-results/admission_trajectory_controller_breakdown_20261002`. The exact physical command and source/input bindings are in the manifest. Launchers guard against overwriting completed results; any deliberate rerun needs a fresh output binding. No giant raw router traces are committed.

Stop here for owner review. No admission redesign, token-load constraint, controller optimization, Coverage tuning, same+path tuning, replication or migration was introduced.
