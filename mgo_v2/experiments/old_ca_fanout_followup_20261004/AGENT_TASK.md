# AGENT TASK — OldCA fan-out follow-up

Do not interrupt the active tolerance_001 B32/decode64 physical study. Start
only after it fully exits and commits.

1. Read README.md and PLAN.md.
2. Port only the old `_incremental_cost` + balanced-Hungarian communication
   objective from commit a0be82e into the current replay/PLAN path as
   `OldCA-fanout`.
3. Add CPU tests proving:
   - same balanced quota slots as BR/Current-CA;
   - exact parity with a0be82e objective on synthetic fixtures;
   - pre-owned destination coalescing is counted correctly;
   - the Hungarian base is not mutated by other incoming assignments.
4. Reuse the existing ShareGPT B32/decode64 frozen candidate space. No new
   capture and no expanded seed range.
5. Search one common R4 and one common R8 workload for BR / Current-CA /
   OldCA-fanout. Prefer <=0.1% fetch/H2D mismatch; bounded fallback <=0.25%.
6. Primary selection objective is BR->OldCA reduction in total per-event
   token->remote-rank fan-out. Record peer-byte and critical-rank reductions as
   separate metrics.
7. Build common frozen-route teacher-forced physical plans for all three
   policies and validate hashes/counters before timing.
8. Env2 only, R4 first on GPUs0,1,4,6 then R8 on GPUs0-7. A3/current 3-A2A
   runtime only.
9. Initial physical screen: BR x1, Current-CA x1, OldCA-fanout x1.
10. If a CA variant gains >=1% TPOT/decode-wall versus BR, allow exactly one
    additional BR+that-policy confirmation pair, subject to reuse rules in
    PLAN.md. No further repeats.
11. Keep controller/Hungarian, detailed counters, compilation and heavy
    monitoring outside MEASURE.
12. Commit results and stop. Do not auto-enable old same/path affinity or A2.
