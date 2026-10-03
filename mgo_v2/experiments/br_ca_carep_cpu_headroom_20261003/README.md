# BR / CA / CA-rep CPU headroom

Main research question:

> In multi-GPU MoE offloading, when inter-GPU communication becomes expensive,
> does miss-expert admission need to move from balanced placement to
> communication-aware placement, and can selective replication add further
> headroom?

This packet is CPU headroom only. Env 1 / Env 2 timing is intentionally deferred.

Fixed terminology:
- **Env 1** = NVSwitch.
- **Env 2** = P2P disabled.
- **BR** = Balanced Random.
- **CA** = Comm-aware Balanced current-demand oracle.
- **CA-rep** = CA plus selective future-popularity oracle replication; replicas
  are ordinary evictable cache entries.

Workloads:
1. **MATH** — reasoning-heavy decode workload.
2. **ShareGPT** — general conversational / everyday serving workload.

Substitution similarity uses the original **SERE-style FineWeb-Edu calibration**,
not a workload-specific MATH calibration.

See PLAN.md and AGENT_TASK.md.
