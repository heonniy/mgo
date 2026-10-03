# Timing-variance hypotheses — 2026-10-04

## Observed facts

The physical study produced identical-computation repeats with a large timing
gap: decode was roughly 605 s versus 466 s. Token hashes, final cache-state
hashes, worker source and no-compilation checks matched. Sampled maximum GPU
temperature was 41 C in both discrepant runs.

S0 of the new diagnostic reports a KVM guest with one visible NUMA node/socket,
192 vCPUs, all GPU CPU-affinity fields spanning 0--191, and no exposed GPU PCI
NUMA node. This describes the **guest view only**.

The CPU expert pool used by the physical worker is pageable and file-backed.
The HEAVY monitor traverses process mappings/PSS while decode is running; prior
scans have taken on the order of tens of seconds.

## Hypothesis table

| ID | Hypothesis | Why plausible | Direct test |
|---|---|---|---|
| H1 | Heavy PSS monitor interferes | Large address-space/page-table traversal overlaps CPU-memory/H2D work | HEAVY vs BOUNDARY S1 |
| H2 | Expert-page residency changes repeats | Pageable file-backed pool can have different resident/page-fault state | NORMAL vs PRETOUCH + fault counters |
| H3 | CPU scheduling varies | Eight ranks historically share CPU 0--191 | fixed disjoint rank affinity |
| H4 | Shared H2D/PCIe/host-memory contention | NUMA=1 does not imply independent DMA/host paths | 1/2/4/8 concurrent 9-MiB H2D |
| H5 | Env2 SHM GPU-pair cost is non-uniform | Physical topology may be hidden; host bridges/resources can differ | empirical all-pair SHM matrix |
| H6 | Host NUMA/hypervisor placement is hidden | KVM may flatten host NUMA/PCl locality | only infer from measured asymmetry unless host data is available |
| H7 | Thermal throttling | Would create run-to-run slowdown | currently low likelihood: both runs max 41 C |
| H8 | Recompile/different computation | Could explain large timing gap | currently screened: no-compile + identical hashes |

## Interpretation rules

1. Do not equate guest-visible one-NUMA with physical one-NUMA.
2. Do not call Env2 PCIe-only; it is the validated P2P-disabled SHM condition.
3. Do not label GPU pairs same/cross NUMA unless host-level evidence exists.
4. File-backed memory does not imply disk I/O. Claim disk/page-cache effects
   only when page-fault/I/O evidence supports them.
5. Do not pick the fastest run. The harness must pass the frozen <=5% repeat
   spread gate before policy timing resumes.
6. If multiple hypotheses remain plausible, report them separately rather than
   assigning a single cause from correlation.
