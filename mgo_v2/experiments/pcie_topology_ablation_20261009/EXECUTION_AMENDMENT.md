# Execution on the PCIe host (2026-10-09)

Base: `7ddc55dff8d8a571a79a7fa415ad16fbc47e78dc`, branch
`codex/pcie-topology-ablation-20261009`.

The owner requests the entire PLAN, repairs of reproducible failures, and a
separate commit after each completed experiment. Prior experiment queues in
AGENTS.md are historical and are not part of this execution.

## Owner-approved host source amendment

The owner explicitly selects **one shared 54-GiB expert source per NUMA node**,
read by both GPUs in that node, instead of four private 54-GiB sources.
The actual node totals are approximately 220 GiB (node 0) and 126 GiB (node 1).
Rank-private full pinning requires 216 GiB plus the existing 128-GiB global
headroom guard; that guard cannot pass with current available host memory.
NUMA-shared sources require 108 GiB of unique physical pages. Each rank maps
and registers its node's same pages; registration does not create new expert
copies. Validate same inode/offsets for each pair, sampled page locality and
full mapping NUMA receipts, direct-pinned H2D correctness, and teardown.
Preserve the 128-GiB global headroom guard, C30 GPU cache and eight reserved P2
slots, native BF16 execution, eviction, routing, and the frozen workload.
Apply the shared source identically to every OURS policy. Report the change
explicitly; do not present timings as an unchanged rank-private experiment.

## Device and phase requirements

Only physical GPUs 0,1,4,5 are allowed. Ranks map to those devices in that order;
groups are A={0,1}/NUMA0 and B={4,5}/NUMA1. Stop only our burn before GPU work.
Never stop unrelated workers, including those on GPUs 2,3,6,7.

For isolated decode: completed routing metadata and PLAN checksums -> completed
forward token all-to-all -> CPU phase rendezvous -> mandatory expert H2D only
-> completed copies on all ranks -> expert compute -> completed compute on all
ranks -> return all-to-all. Use a separate Gloo CPU group for phase rendezvous:
an early finisher's NCCL barrier would otherwise overlap another rank's H2D.
Keep the selected production default unchanged. Instrument a separate
diagnostic pass, and preserve all unprofiled repeats.

The owner additionally requires controller and decision work in the C++ engine
to minimize Python overhead. The opt-in native engine owns demand aggregation,
quota rotation, BR/Near/CA assignment, Gate eviction, admission, owner state,
and routing/accounting. The array adapter and experiment orchestration stay in
Python. Gate G1 compares all outputs and residency arrays against the reference,
including deterministic BR randomness and Near/Hungarian tie decisions.

Maintain our burner on GPUs 0,1,4,5 whenever GPU experiments are inactive.
An ownership-checked lease stops our burner before a GPU experiment and restores
it after workers exit, including failed attempts. Downloads and CPU checks do
not hold that GPU lease.

## Paths and bounds

Model: `/data2/esjung/models/Qwen3-30B-A3B-Instruct-2507`.
Dataset and byte-identical frozen manifests: `/data2/esjung/datasets/`.
Conda environment: `/data2/esjung/envs/mgo-pcie`.
CUDA build tools: `/data2/esjung/envs/cuda121`.
Raw outputs: `/data2/esjung/mgo-results/pcie_topology_ablation_20261009/`.

Pre-register 30 timed samples per microbench cell after five warmups,
randomized paired cell order with seed 20261009, and three unfiltered primary
repeats per policy with counter-ordered arms. A maximum relative TPOT spread
above 5% triggers exactly two additional repeats for every comparable arm.
Keep every attempt. Each model job has a 3600-second bound; external baseline
jobs have a 14400-second bound. Respect a STOP file in the raw-output root.
Failures are repaired with a new attempt directory and receipt rather than
overwriting evidence or silently changing the workload/cache budget.
