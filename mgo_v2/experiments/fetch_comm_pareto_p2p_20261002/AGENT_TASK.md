## Completed — IPC baseline rebase and I1 communication gate

**IPC baseline rebase complete (2026-10-03): AMBIGUOUS_GAP.**
Both NCCL_CUMEM_ENABLE=0 smokes passed: T0 uses P2P/IPC only and R3
uses SHM/direct/direct only, on one node/four local ranks. The original
384-event communication replay passed all payload checks in both orders.
Primary R3/T0 ratios are 1.0529x and 1.0836x (median 1.0682x), below the
1.20x STRONG_GAP threshold. CPU Pareto results and frozen schedule metadata
are SHA256-identical; no CPU screen or model was run. Peak process-tree RSS
was 2.72 GiB with no OOM. GPUs 0,1,4,5 are released. Read
`IPC_BASELINE_REBASE_RESULTS.md` and `ipc_baseline_validation.json` in the
experiment packet. Stop here; do not launch clean F/K, add repetitions or
tune transport. Earlier checkpoints below are historical.

## Immediate task — adopt IPC T0 baseline and rerun E1

Current checkpoint: `f64eb56`.

Read `IPC_BASELINE_REBASE.md`.

Do exactly:

1. keep the CPU Pareto/F-K selection unchanged;
2. set NCCL_CUMEM_ENABLE=0 for both new transport modes;
3. verify T0-IPC uses P2P/IPC and R3-SHM uses SHM with no P2P/NET;
4. if both paired smokes pass, re-run the frozen 384-event E1 trace replay;
5. commit the transport rebase/E1 result and stop.

Only if E1=STRONG_GAP may the already-planned clean F/K model timing proceed.

Do not use default P2P/CUMEM again, rerun CPU simulation, tune NCCL, or rerun C.

## Completed — bounded cuMem diagnosis

**Bounded cuMem diagnosis complete (2026-10-03): CUMEM_PATH_UNSTABLE.**
All three fresh default-T0 trials selected P2P/CUMEM and timed out at the
90-second bound. The single NCCL_CUMEM_ENABLE=0 diagnostic selected P2P/IPC
and passed four-rank payload validation. This supports path-specific
instability, not an established driver root cause. No E1 retry or model
work was started; the alternate path is not adopted as T0. Target GPUs
0,1,4,5 were released; other GPU jobs were not modified. Read
`CUMEM_PREFLIGHT_RETRY_RESULTS.md` and `cumem_preflight_retry.json`.
Stop here: do not add retries, change the timed baseline or reset drivers.

Earlier instructions below are historical.

## Immediate task — bounded CUMEM diagnosis and conditional E1 retry

Current checkpoint: `b71199c`.

Read `CUMEM_PREFLIGHT_RETRY.md`.

Do exactly:

1. snapshot current GPU/process/environment state;
2. run default T0 32-KiB preflight in three fresh process groups;
3. run one fresh T0 diagnostic with `NCCL_CUMEM_ENABLE=0`;
4. commit the diagnosis immediately;
5. only if default T0 passes 3/3 on P2P/CUMEM, re-run the original E1 measurement unchanged in a fresh result root and commit E1 separately.

If any default T0 trial is unstable, do not run E1 and do not use the cuMem-disabled path as a substitute baseline.

No model/E2 unless the re-run E1 later satisfies the original STRONG_GAP gate.

## Blocked — E1 preflight

**E1 trace-communication stage blocked (2026-10-03): BLOCKED_PREFLIGHT.**
The first T0 32-KiB preflight exceeded its 180-second bound. Communicators
initialized and selected P2P/CUMEM; no validated payload receipt followed.
Last logs show shareable-buffer imports/UDS mapping; root cause is not
established. No OOM: peak process-tree RSS 4.69 GiB, minimum host available
1851.92 GiB. Target GPUs were released. The 384-event input/count validation
and two CPU tests pass, but zero trace timings or model runs were executed.
Read `TRACE_COMM_REPLAY_RESULTS.md` and `trace_comm_preflight_failure.json`.
Do not label this NO_GAP: no ratio exists. E2 remains unauthorized by its
STRONG_GAP gate; stop without an automatic retry, NCCL tuning or server move.

Earlier instructions below are retained as history.

## Immediate task — actual-trace comm replay, then conditional clean F/K

Current checkpoint: `b6edeb0`.

Read `TRACE_COMM_CLEAN_FK.md`.

Stage E1 is mandatory and model-free:

1. replay the validated 384-event decode dispatch/combine traffic exactly;
2. compare T0 vs R3 in two counter-ordered passes;
3. commit E1 immediately.

Do **not** start a model unless E1 satisfies STRONG_GAP exactly as defined.

If E1=STRONG_GAP, then:

4. remove heavy replay/check validation from the timed path;
5. run only F=rho0 and K=rho.25, two repeats per T0/R3 (8 cells);
6. commit E2 and stop.

No C, extra rho, longer decode, Nsight, substitution, weighted method or further NCCL tuning.

## Completed — six-cell physical F/K/C pilot

**Physical F/K/C pilot complete (2026-10-03): NO_CLEAR_SHIFT.**
All six cells passed trace/cache/action/send-count/token parity (2,592 global
layer events). Descriptive winners differ: T0-K (8.3423 s decode) and R3-F
(6.8918 s), with 5.19%/9.30% margins. However, R3/T0 payload intervals are
0.769x at F and 0.866x at K, and F replay/check CPU time differs by 0.991 s;
the proposed communication-cost mechanism is not supported. No repeat or
follow-up is authorized by this result. Read `PHYSICAL_FKC_RESULTS.md` and
`physical_fkc_validation.json` in the packet; stop for owner review.

The instructions below are retained as history.

## Immediate task — six-cell physical F/K/C pilot

Current checkpoint: `dc7b099`.

Read `PHYSICAL_FKC_PILOT.md`.

Do only this:

1. regenerate and hash frozen F/K/C action schedules from the validated CPU replay;
2. add the narrow experiment-only physical replay path needed to apply those schedules;
3. run exactly six short cells in the prescribed order:
   T0-F, R3-C, T0-K, R3-K, T0-C, R3-F;
4. require exact trace/cache/send-count/token parity;
5. report decode time plus physical H2D/peer mechanism counters;
6. commit results and stop.

No substitution, no policy retuning, no more rho values, no automatic repeats,
no Nsight, no 32/64-step run, no final weighted method.

## Completed — CPU-only replica screen

**GO_FOR_OWNER_REVIEW**, all five rho points nondominated; F/K/C=0/.25/.75.
Decode H2D +160.20%, peer activation bytes -100% from F to C. All 2,160
CPU replay events passed, with independent rho=0 parity at all 432 events.
See [REPLICA_PARETO_RESULTS.md](REPLICA_PARETO_RESULTS.md). Stop after
committing this result. No new GPU/model run or physical F/K/C is authorized
by completion of the gate. The tasks below are historical.

## Immediate task — CPU-only replica Pareto screen

Current checkpoint: `ed7f82b`.

Transport characterization is complete enough for this decision. Do not run more NCCL tests.

Read `REPLICA_PARETO_SCREEN.md` and:

1. reuse the validated exact-only 8-decode capture;
2. implement only the deterministic CPU replay described there;
3. sweep rho = 0, 0.125, 0.25, 0.50, 0.75;
4. report decode H2D bytes versus peer activation bytes and nondominated points;
5. commit immediately and stop.

No new model generation, GPU timing, replication runtime, substitution, load objective, final weighted loss or Stage 3.

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
