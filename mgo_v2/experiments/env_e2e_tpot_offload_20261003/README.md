# Env 1 / Env 2 physical offloading E2E + TPOT

Purpose: physically validate the CPU-headroom result with **real GPU expert
offloading**, while preventing compilation, controller, or instrumentation
overhead from contaminating timing.

Terminology:
- **Env 1** = NVSwitch P2P/IPC.
- **Env 2** = P2P disabled, validated SHM path.
- **BR** = Balanced Random.
- **CA** = Comm-aware Balanced.
- **CA-rep** = CA + selective replica admission.

Primary system uses **Gate eviction + substitution ON**. A small exact-only
control isolates placement from substitution. LRU is not rerun physically
unless the predeclared fallback gate is triggered; it was already swept in the
completed CPU study.

See PLAN.md and AGENT_TASK.md.
