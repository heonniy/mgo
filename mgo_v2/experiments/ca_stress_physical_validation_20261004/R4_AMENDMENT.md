# R4 extension — best CA stress cell

Date: 2026-10-04
Status: owner-authorized extension.

Do **not** interrupt or mutate the already-running R8-best physical validation.
Finish and commit that bounded R8 run first. Then run this R4 extension.

## Frozen R4-best cell

Source manifest:
`../ca_stress_workload_search_20261004/selected_manifests/ShareGPT_R4_B8_c30.json`

Settings:
- dataset: ShareGPT
- R=4
- physical GPUs: 0,1,2,3 for every phase
- local B=8 (global N=32)
- cache=30% global
- Gate W128
- substitution OFF
- decode256
- sample_seed=205
- dp_seed=4
- BR seed=73
- policies: BR and CA
- environments: Env 1 and Env 2

Do not change the 4-GPU subset after timing results.

## Frozen CPU resource expectation

From commit `0b6da6e`:
- CA peer bytes: 14,646,489,088
- BR(seed73) peer bytes: 20,957,196,288
- chosen-seed BR->CA peer reduction: **30.1124%**
- audited eight-seed median reduction: **29.8593%**
- audited seed range: **29.6355%--30.1124%**

The R4 stress effect is therefore not created by one pathological BR seed.

## Measurement

Reuse exactly the stabilized harness:
- BOUNDARY-only safety monitoring during timed execution
- fixed disjoint CPU affinity
- NORMAL expert residency behavior
- PLAN -> COMPILE/WARMUP -> MEASURE -> COUNTERS
- Hungarian/controller work only in PLAN
- no PSS/smaps, profiler, detailed counters or compilation in MEASURE
- same outer-boundary synchronization and hash checks

For each environment run BR and CA with 3 clean MEASURE repeats.
Counterbalance:
1. Env1 BR -> Env1 CA -> Env2 CA -> Env2 BR
2. Env2 BR -> Env2 CA -> Env1 CA -> Env1 BR
3. Env1 CA -> Env1 BR -> Env2 BR -> Env2 CA

If either policy in one environment has >5% spread, run exactly two additional
repeats for that implicated pair only.

Required output:
- E2E / TPOT / decode-wall median and full range
- BR->CA timing gain in Env1 and Env2
- separate COUNTERS peer/H2D/hit/eviction/reload/quota metrics
- route/action/token/final-cache hashes
- no-compile and physical-cache validation

Compare the physical R4 result with the R8-best result, focusing on whether the
larger resource reduction (~30% vs ~21%) produces a larger timing effect.

Label both as optimized communication-stress workloads, not dataset averages.
Commit the R4 extension results and stop. No additional cells are authorized.
