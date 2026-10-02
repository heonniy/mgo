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
