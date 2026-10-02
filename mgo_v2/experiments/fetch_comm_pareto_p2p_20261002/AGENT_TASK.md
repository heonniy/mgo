## Immediate task — one exact-only payload capture, then crossover

Current checkpoint: `0c09fae`.

The prior existing-trace audit is complete and must not be repeated.

Read `EXACT_PAYLOAD_CAPTURE.md` and run only:

1. one R4/B8 P0 exact-only capture (substitution off, replication off, LRU, 1 prefill + 8 decode forwards, T0);
2. extract actual dispatch/combine rank-pair payload p50/p90/p99/max from recorded runtime counts;
3. if capture validation passes, run the five-size T0/R3 crossover with two counter-ordered passes;
4. commit results immediately and stop.

Do not implement replication, run Stage 1, add quality evaluation, or search more NCCL knobs.

## Immediate task — recover T1 transport only

Current checkpoint: `7c881f7`. Do **not** implement replication or run the model yet.

Read `TRANSPORT_RECOVERY.md` and execute only its bounded R1 -> R2 -> R3 smoke sequence. Commit after every success/failure.

Important:

- R1 uses `NCCL_P2P_LEVEL=LOC` with `NCCL_P2P_DISABLE` unset.
- R2 additionally disables GDR.
- R3 disables IB and accepts Socket as an explicit synthetic slow-communication condition.
- Stop at the first passing non-P2P condition.
- If all bounded conditions fail, stop H100 synthetic T1 work and report `BLOCKED_SYNTHETIC_TRANSPORT`.

After the first success, run only the three tiny calibration cells. Resume Stage 1 only after committing that result.

# AGENT TASK — Fast Fetch/Comm Pareto characterization

Read `PLAN.md` and the completed controller-overhead results at `../controller_overhead_20261002/RESULTS.md`.

Implement only the minimum experiment in PLAN.md.

Hard constraints:

- R4 on physical GPUs 0,1,4,5.
- Qwen3-30B-A3B-Instruct-2507 BF16.
- B8 only, 32 decode forwards.
- substitution **off**;
- LRU fixed;
- no load-aware objective;
- replication is the only new residency knob;
- a replica is loaded from CPU through the normal H2D expert path;
- compare normal NVSwitch against `NCCL_P2P_DISABLE=1` with SHM left enabled;
- CPU sweep first, then only F/K/C on GPU;
- maximum 12 primary generations unless the >10% repeat-spread rule triggers one targeted repeat;
- do not implement the final weighted/Pareto method yet.

Transport verification is mandatory. Do not label T1 as PCIe-only. Save one INFO transport log for T0 and T1, then disable INFO logging for timing.

Stop and summarize after the bounded matrix.
