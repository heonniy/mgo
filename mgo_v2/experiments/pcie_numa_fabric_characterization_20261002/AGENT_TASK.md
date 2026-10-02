# AGENT TASK — PCIe/NUMA fabric characterization

Read README.md, PLAN.md, matrix.json, then ../../AGENTS.md and existing fabric benchmark code.

Use only:
- rank0 -> GPU1 -> NUMA0
- rank1 -> GPU2 -> NUMA0
- rank2 -> GPU3 -> NUMA1
- rank3 -> GPU5 -> NUMA1

Each worker must bind CPU+memory before CUDA import and before allocating/pinning H2D host buffers. Wrong-NUMA memory invalidates the run.

The server is shared. Do not stop or alter unrelated jobs. Record background load and keep the benchmark bounded.

Implement/extend a small fabric benchmark to measure:
1. isolated/same-NUMA/cross-NUMA/four-way 9 MiB H2D;
2. PXB vs SYS pairwise GPU communication;
3. four-rank NCCL all-to-all;
4. H2D+NCCL overlap contention.

Record CUDA P2P capability and actual NCCL transport; never assume direct P2P.

Publish TOPOLOGY.md, manifest/binding receipts, background-load logs, H2D/P2P/NCCL/overlap CSV+JSON, RESULTS.md and provenance hashes.

Stop after characterization. Do not change cache/admission policy automatically.
