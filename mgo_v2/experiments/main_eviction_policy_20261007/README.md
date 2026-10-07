# MAIN eviction policy packet

Prepared from the current H0/full-pinned LA_CA_NEAR runtime after the BR cache
audit reported low MAIN distinct-expert reuse.

Files:
- `PLAN.md`: frozen question, semantics and safety bounds.
- `scripts/main_eviction_replay.py`: deterministic P0 MAIN-only replay.
- `scripts/run_main_eviction_policy_study.py`: bounded B8/B16/B64 capture+replay.
- production code changes are opt-in trace instrumentation only.

No GPU result is included by this preparation commit.
