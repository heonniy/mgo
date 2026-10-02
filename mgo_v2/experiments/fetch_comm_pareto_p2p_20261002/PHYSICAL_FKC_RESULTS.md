# Physical F/K/C pilot

**NO_CLEAR_SHIFT**. Completed correctness-passing cells: 6/6.

| Cell | Status | Prefill s | Decode s | Mean decode ms | E2E s |
|:---|:---|---:|---:|---:|---:|
| T0-F | PASS | 6.7977 | 8.7989 | 1099.8563 | 15.5938 |
| R3-C | PASS | 6.6108 | 9.0642 | 1133.0310 | 15.6753 |
| T0-K | PASS | 7.6258 | 8.3423 | 1042.7905 | 15.9679 |
| R3-K | PASS | 5.4909 | 7.5982 | 949.7755 | 13.0891 |
| T0-C | PASS | 5.6232 | 9.2553 | 1156.9172 | 14.8708 |
| R3-F | PASS | 5.6235 | 6.8918 | 861.4766 | 12.5153 |

Descriptive winners are T0-K and R3-F; margins over the next-best point are 5.19% and 9.30%, respectively. One run per cell provides no confidence interval or stable speedup claim.

Frozen application and logical checks contribute 1.874–2.865 seconds per decode at the maximum rank. These costs are included in the primary times; differences cannot be attributed to transport alone.

The different-winner and >=5% margin checks pass, and physical H2D/peer counts follow the CPU frontier. However, the proposed slower-R3 communication mechanism is not supported: R3/T0 payload collective intervals are 0.769x at F and 0.866x at K. The same F schedule also has 0.991 seconds more application/check time in T0 than R3. These single-run differences confound a transport-driven interpretation, so the plan's component-support clause yields NO_CLEAR_SHIFT. No overhead subtraction or timing correction is applied. Intervals include self traffic, waits and CPU/launch gaps; they do not isolate wire transfer time.

| Cell | Decode H2D GiB | Decode peer MiB | Apply CPU ms (max rank) | Payload collective interval ms (max rank) |
|:---|---:|---:|---:|---:|
| T0-F | 154.9951 | 319.4531 | 2865.206 | 186.159 |
| R3-C | 403.2949 | 0.0000 | 1943.218 | 70.096 |
| T0-K | 219.0762 | 185.2969 | 2045.506 | 149.661 |
| R3-K | 219.0762 | 185.2969 | 1941.784 | 129.658 |
| T0-C | 403.2949 | 0.0000 | 1975.792 | 69.896 |
| R3-F | 154.9951 | 319.4531 | 1874.160 | 143.176 |

## Frozen scope and validation

- F/K/C rho=0/.25/.75; 1 prefill + 8 decode forwards; physical GPUs 0,1,4,5 only. No substitution, migration, policy search in timing, full-expert D2D copies, extra rho, repeats or profiler.
- Three immutable schedules match the previous CPU screen and passed all 1,296 action/state comparisons. Two CPU codec tests pass, including rejection of corrupted actions/state.
- Transport preflights: [{'mode': 'T0', 'status': 'PASS'}, {'mode': 'R3', 'status': 'PASS'}]. INFO is confined to preflight, disabled in model cells. Runtime environments are checked exactly.
- Completed physical events check raw expert IDs/origins, copy sets, received and returned expert routes, physical fetch increments, and dispatch/combine counts. A completed cell additionally checks every cross-rank transpose and generated tokens against the source.
- Timings include obligatory per-event validation, action application and existing lightweight CUDA interval instrumentation. Decode is maximum cumulative rank duration; loading, schedule generation and transport preflight are excluded.
- Native fetch-wait/compute counters are host timings, not isolated GPU H2D/kernel durations; the latter are omitted. Interval sums are not additive critical-path components.
- Memory guard: minimum host available 1613.06 GiB; peak process-tree RSS 227.08 GiB; minimum target GPU free 70,323 MiB.
- Frozen implementation/schedule commit: `835a03552f6bde49a1d20af60aa87fb6d9639d60`. Raw receipt hashes are in `physical_fkc_pilot.json`.

See [execution conventions](PHYSICAL_FKC_EXECUTION.md), [schedule metadata](physical_fkc_schedule_metadata.json), [CSV](physical_fkc_pilot.csv), [full result](physical_fkc_pilot.json), and [validation](physical_fkc_validation.json).

Stop for owner review. Do not automatically retry, extend the trace, tune rho/transport, or start a different server experiment.
