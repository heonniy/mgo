# PLAN — Dual-NUMA PCIe communication and contention

Date: 2026-10-02
Status: plan only; do not launch from this commit.

## 1. Fixed GPU/NUMA mapping

| Logical rank | Physical GPU | NUMA |
|---:|---:|---:|
| 0 | 1 | 0 |
| 1 | 2 | 0 |
| 2 | 3 | 1 |
| 3 | 5 | 1 |

Expected topology:
- GPU1<->2: same-NUMA PXB
- GPU3<->5: same-NUMA PXB
- cross-group pairs: SYS

No automatic GPU substitution.

## 2. Strict local host fetch

Bind CPU and memory policy before CUDA import and before allocating host buffers.

Required:
- GPU1/2 read only NUMA0-local pinned host pages.
- GPU3/5 read only NUMA1-local pinned host pages.

Verify physical GPU identity, PCI NUMA node, CPU affinity and memory policy. Sample page residency with move_pages/equivalent when permitted. Fail before timing on any mismatch.

## 3. Shared-server protocol

Unrelated jobs may remain active. Never stop, pause or reconfigure them.

Before/after every block record:
- nvidia-smi utilization/memory for GPUs 1,2,3,5
- CPU load
- NUMA memory availability
- timestamp

Keep blocks short, retain every repeat, and report median/min/max plus background load. These are contention-inclusive observations, not isolated hardware peaks.

## 4. Stage 0 — topology/capability

Save:
- nvidia-smi topo -m
- nvidia-smi topo -p2p p if supported
- lspci -tv
- PCIe LnkCap/LnkSta for selected GPUs/upstream bridges
- CPU/NUMA topology
- CUDA peer-access matrix
- one NCCL topology/debug dry run

Explicitly state whether same-NUMA and cross-NUMA pairs have direct CUDA P2P. Do not assume direct GPU P2P.

## 5. Stage A — local H2D contention

Primary payload: 9 MiB, matching one validated mgo expert.
Secondary short saturation control: 64 MiB.

A1 isolated:
- GPU1<-NUMA0
- GPU2<-NUMA0
- GPU3<-NUMA1
- GPU5<-NUMA1

A2 same-NUMA concurrent:
- GPU1+2, both NUMA0-local
- GPU3+5, both NUMA1-local

A3 cross-NUMA concurrent:
- GPU1+3
- GPU2+5
Each GPU still reads only its local NUMA memory.

A4 four-way:
- GPU1,2 from NUMA0
- GPU3,5 from NUMA1

Report per-GPU latency/GBps and aggregate GBps.

Compute:
- per-GPU slowdown = isolated BW / concurrent BW
- aggregate efficiency = sum(concurrent BW) / sum(isolated BW)
- fairness/CV

Use short warmup and >=10 samples.

## 6. Stage B — GPU-to-GPU PCIe communication

Pair classes:

Same NUMA:
- 1<->2
- 3<->5

Cross NUMA:
- 1<->3
- 1<->5
- 2<->3
- 2<->5

Payloads:
- 4 KiB
- 64 KiB
- 1 MiB
- 4 MiB
- 9 MiB
- 32 MiB

If CUDA P2P is supported, measure peer-copy one-way and bidirectional bandwidth. Always measure NCCL send/recv because mgo uses NCCL.

Report actual transport/path, latency, payload GBps and same-NUMA vs cross-NUMA ratios.

## 7. Stage C — 4-rank NCCL

Use exactly GPUs 1,2,3,5.

Measure all-to-all latency/bandwidth for per-rank payloads:
- 1 MiB
- 4 MiB
- 32 MiB

Report slowest-rank interval and per-rank spread.

## 8. Stage D — H2D + GPU communication co-contention

This is the offloading-relevant stage.

Compare:
1. H2D only
2. NCCL only
3. overlapped H2D + NCCL

Groups:
- same-NUMA pair 1+2
- cross-NUMA pair 1+3
- all four 1+2+3+5

H2D payload = 9 MiB local expert copy.

Report:
- H2D interference = overlapped H2D time / H2D-alone time
- communication interference = overlapped comm time / comm-alone time

This answers whether expert fetch and GPU communication contend for the same PCIe/root/UPI resources.

## 9. Interpretation

Keep three questions separate:
1. Does same-NUMA concurrent H2D collapse per-GPU bandwidth?
2. How much more expensive is cross-NUMA SYS GPU communication than same-NUMA PXB?
3. Does NCCL communication slow local expert H2D, or vice versa?

Because this is a shared server, do not present values as uncontended hardware limits.

## 10. Required artifacts

Create:
- TOPOLOGY.md
- measurement_manifest.json
- numa_binding_receipts.json
- background_load.csv/json
- h2d_contention.csv/json
- p2p_pair_matrix.csv/json
- nccl_4rank.csv/json
- overlap_contention.csv/json
- RESULTS.md
- provenance/hash receipts

Do not commit large traces.

## 11. Stop

Stop if:
- any GPU/NUMA mapping is wrong
- a host buffer violates local memory policy
- another user's process would need modification
- the selected GPU set changes

Stop after characterization; do not automatically tune topology-aware admission.
