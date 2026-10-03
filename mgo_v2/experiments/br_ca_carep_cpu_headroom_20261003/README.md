# Completed results

See [RESULTS.md](RESULTS.md), [validation.json](validation.json), and
[EXECUTION.md](EXECUTION.md). Both decode64 and decode256 were authorized by
the owner and completed: 1536 main + 16 BR seed-audit CPU replays.
Two 512-request master captures used eight GPUs in one loading session.
FineWeb-Edu SERE similarity was hash-verified and reused; no recalibration.

CA_HEADROOM: 283/512; CA_STRONG_HEADROOM: 33/512. CA-rep reached neither
20% headroom nor tradeoff criterion. These are frozen-route resource results,
not quality or Env timing. CPU peak aggregate RSS 6.64 GiB; all GPU workers
exited successfully. Stop for owner review; no automatic follow-up.

---

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

Each dataset is captured once for **256 decode tokens** on 8 GPUs. The
**64-token horizon is the exact first 64 steps of that same capture**, so the
64-vs-256 trace comparison requires no extra generation run.

Substitution similarity uses the original **SERE-style FineWeb-Edu calibration**,
not a workload-specific MATH calibration.

See PLAN.md and AGENT_TASK.md.
