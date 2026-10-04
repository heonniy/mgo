# PLAN — pinned H2D + CoSLoT + active-peer

## Goal
Test whether BR->CA TPOT was masked by pageable H2D, fixed all-to-all alpha,
and severe CPU->GPU expert-fetch pressure.

The requested follow-up disables expert substitution. This makes the cache
capacity experiment interpretable as pure exact-cache/offloading pressure:
no miss route can be rescued by a similar resident expert.

## PLAN provenance
PLAN identity is keyed by **cache ratio and substitution mode**. A PLAN path is
not a required input.

For each BR/CA policy:
1. discover a launcher PASS + schedule_validation PASS PLAN with the requested
   cache ratio and substitution flag;
2. if absent, generate a current/pageable PLAN outside timing;
3. validate it;
4. freeze it;
5. reuse that exact schedule for all transport/H2D cells.

A c30/s1, c30/s0, c60/s1, and c60/s0 PLAN are distinct artifacts. No
transport-specific replanning is allowed.

## Stage A — normalize H2D
Primary measurements use two pinned expert-sized staging buffers, a dedicated
H2D stream, async DMA, fetch-event compute gating, and compute-event overwrite
protection, following the CoSLoT/Archer mechanism.

## Stage B — same PLAN, three transports
For each cache arm, run BR and CA with:
1. current + pinned
2. coslot + pinned
3. coslot-active + pinned

Each cell uses COMPILE x1, MEASURE x3, COUNTERS x1.

## Stage C — requested 60% cache arm, substitution OFF
Primary follow-up:
- workload: R = MATH / R4 / local B64 / Gate eviction;
- cache ratio: **0.60**;
- substitution: **OFF**;
- policies: BR, CA;
- environment: env1 first;
- transports: current, coslot, coslot-active;
- H2D: pinned.

The completed CPU replay for the matching substitution-OFF condition gives the
reason for this arm:
- c30 BR exact-global hit: ~59.9%, residual miss: ~40.1%, H2D: ~7.16 TiB;
- c60 BR exact-global hit: ~86.5%, residual miss: ~13.5%, H2D: ~3.26 TiB;
- BR peer traffic remains ~104.5 GiB at both cache sizes.

So c60 cuts H2D by about 54% while leaving raw peer demand nearly flat. This is
a cleaner regime for asking whether communication becomes visible in TPOT once
fetch pressure falls.

Note that CA's peer-byte advantage is itself smaller at c60 in the CPU replay
(~100.7 vs 104.5 GiB) than at c30 (~97.2 vs 104.5 GiB). Therefore this is not
predeclared to improve CA TPOT; the experiment tests which effect dominates:
less H2D masking versus less communication-placement headroom.

For a direct cache-pressure comparison, rerun both c30/s0 and c60/s0 with the
same transport matrix.

## Env2
Env2 is an HGX/H100 machine with GPU P2P disabled and SHM forced. It is a
transport stress/control, not a genuine weak PCIe/NUMA machine.

## Hypotheses
- Pinned H2D removes an implementation artifact from expert fetch.
- c60/s0 substantially lowers exact miss pressure relative to c30/s0.
- CoSLoT tests whether fewer collective rounds beat lower byte volume.
- Active-peer tests whether CA fan-out reduction can reduce alpha, not only beta.
- If BR->CA remains small under c60/s0 + pinned + active-peer, inter-GPU
  communication is unlikely to be the dominant remaining critical path.

## Required report
For each cache/policy/transport report median TPOT/E2E plus:
- exact-global hit and residual miss fraction from the frozen PLAN;
- H2D bytes;
- activation peer bytes and total wire bytes;
- A2A call count;
- active-peer batch count and P2P op count;
- zero-remote rounds;
- mean/max active-peer degree.

Primary comparison: BR vs CA under **c60/s0 + coslot-active + pinned**.
Secondary comparison: c30/s0 vs c60/s0 to measure how the bottleneck shifts as
offloading pressure falls.
