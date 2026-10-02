# PCIe / NUMA fabric characterization — 4 GPUs, no NVLink — 2026-10-02

This experiment characterizes the dual-NUMA PCIe-only server for multi-GPU MoE offloading.

Use exactly physical GPUs 1,2,3,5:
- GPU1, GPU2 -> NUMA0
- GPU3, GPU5 -> NUMA1

Topology classes:
- same-NUMA PXB: (1,2), (3,5)
- cross-NUMA SYS: every pair across {1,2} and {3,5}

The server is shared and the owner permits short overlap with unrelated GPU jobs. Do not stop or modify those jobs. Record background load and label all results shared-server/contention-inclusive.

Non-negotiable rule: every GPU fetches host expert bytes only from pinned pages allocated on its matched NUMA node. Wrong-NUMA host allocation invalidates the run.
