# Execution amendment

Owner selected d6be61d on codex/policy-regime-20261005 as the active goal.
Previous three-arm H2D profiling was cooperatively stopped; its two completed
captures remain in the refactoring worktree/raw output, not pooled here.

Use only B128, decode64, R4 physical 0/1/4/5, BF16, fixed staging CPU team,
async frozen metadata, unique combine, V3 P2/T2. Reuse identical existing input
bytes by read-only use of symlinks; regenerate policy proofs for C30 and C60.
No new trace or model inference inputs. One physical arena per cache group.
Order: BR/OLD_CA/FCA/LA_CA then reverse. Two stable rounds stop; at most three
rounds if the timing stability gate requests it. No five-repeat extension.

OLD_CA id3 is now supported by the prefetch controller. Its prefetch assignment
uses demand-locality, as FCA/CA already do, since the frozen predictor has no
future token-level co-routing. Mandatory admission remains historical static
fanout. LA_CA retains existing LA prefetch placement. This fallback is explicit;
we do not fabricate future token packet masks.

Noise reporting: preserve every raw sample and paired gain. Do not exclude a
sample solely for high latency, unfavorable gain, or being the second repeat.
An exclusion requires independent technical evidence (wrong affinity, recompilation,
resource interference, invalid workload/copy invariants), with its reason and raw
value retained. If three valid samples differ without independent evidence, report
the median and full range, label instability, and do not claim noise-free TPOT.
Clean two-repeat summaries use mean and full range. All timing remains unprofiled;
mechanism profiling is a separate later phase.

Validation before GPU: 16 CPU replays (two capacities, four policies, P0/P2)
passed. Independent packet-set reconstruction checks incremental incident counts,
packet cost, and balanced quota for FCA/LA_CA across 12 fixtures.
