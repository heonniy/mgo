# PLAN — pinned H2D + CoSLoT + active-peer

## Goal
Test whether BR->CA TPOT was masked by pageable H2D and fixed all-to-all alpha.

## PLAN provenance
PLAN path is not a required input. For each BR/CA policy:
1. discover a launcher PASS + schedule_validation PASS PLAN;
2. if absent, generate current/pageable PLAN outside timing;
3. validate it;
4. freeze it;
5. reuse the exact schedule for all transport/H2D cells.
No transport-specific replanning is allowed.

## Stage A: normalize H2D
Primary measurements use two pinned expert-sized staging buffers, dedicated H2D
stream, async DMA, fetch-event compute gating, and compute-event overwrite
protection, following CoSLoT/Archer's mechanism.

## Stage B: same PLAN, three transports
R/env1 BR and CA:
1. current + pinned
2. coslot + pinned
3. coslot-active + pinned
Each: COMPILE x1, MEASURE x3, COUNTERS x1.
If stable, repeat the exact matrix on env2.

## Env2
Env2 is an HGX/H100 machine with GPU P2P disabled and SHM forced. It is a
transport stress/control, not a genuine weak PCIe/NUMA machine.

## Hypotheses
- Pinned H2D reduces unrelated masking overhead.
- CoSLoT tests whether fewer collective rounds beat lower byte volume.
- Active-peer tests whether CA fan-out reduction can reduce alpha, not only beta.
- If BR->CA is still small after both corrections, communication is not the
  dominant critical path in this regime.

## Required report
Median TPOT/E2E; BR->CA delta; H2D bytes; peer/wire bytes; A2A call count;
active-peer batch count; P2P op count; zero-remote rounds; mean/max active-peer
degree. Primary result: BR vs CA under coslot-active+pinned.
