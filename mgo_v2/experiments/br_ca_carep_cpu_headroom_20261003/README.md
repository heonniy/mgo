# BR / CA / CA-rep CPU headroom

This is the new owner-priority experiment for the main research question:

> In multi-GPU MoE offloading, when inter-GPU communication becomes expensive,
> does miss-expert admission need to move from balanced placement to
> communication-aware placement, and can selective replication add further
> headroom?

This packet is **CPU headroom only**. It does not time Env 1 or Env 2.

Policy names are fixed:
- **BR**: Balanced Random.
- **CA**: Comm-aware Balanced current-demand oracle.
- **CA-rep**: CA plus one future-popularity-based optional replica per miss
  expert; replicas are ordinary cache entries and may be evicted normally.

The workload is a decode-heavy MATH workload with one master trace that is
logically repacked into all R/batch settings.

See PLAN.md and AGENT_TASK.md.
