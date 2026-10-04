# Owner timing repetition amendment

Applies immediately to the active R8 and both queued R4 GPU-set studies.
Supersedes all earlier three-repeat, 5%-spread and two-extra-repeat schedules.
Completed historical studies are not rerun.

For each (cell, environment, policy), collect exactly two clean MEASURE samples.
For E2E and TPOT separately compute abs(T1-T2)/mean(T1,T2).

- Both differences <=2%: stop at two. Report mean, median and full range.
- Maximum difference >2% and <=5%: run exactly one third sample and use median.
- Either difference >5%: stop at two, mark unstable, no primary timing claim.

The greater-than-5% rule takes priority if the other metric is in the middle
band. Decode wall remains descriptive and does not trigger extra repeats.
A third-sample outlier (>5% pairwise E2E/TPOT difference) is flagged in the
final report without adding another run. Preserve all samples; never select
only the closest two. No single-sample primary BR-vs-CA result is allowed.
If either policy is unstable, its environment's displayed gain is descriptive
only and primary_comparison_valid is false.

Use the original counterbalanced orders for repeats1 and2. Schedule only
qualifying cells for repeat3, filtering the original order Env1 CA, Env1 BR,
Env2 BR, Env2 CA. Decisions are independent per policy; do not force a third
for its counterpart. This gives 8--12 MEASURE generations per configuration,
24--36 across R8 and the two R4 sets. PLAN, COMPILE/WARMUP, COUNTERS, workload,
seed, safety guards and output/hash validation are unchanged.

Queue order stays R8 -> R4 GPUs0,1,4,6 -> R4 GPUs0,1,2,3. Restore all eight
resident model workers after the queue completes. No subsequent sweep.

At amendment time R8 was still in CA PLAN and had no MEASURE samples. Its
coordinator was paused while the original PLAN workers completed, with safety
checks maintained by the handoff controller. All eight PLAN receipts passed;
the old coordinator was retired only after the PLAN worker group exited.
The updated driver reuses both completed PLANs and continues without repeating
scientific work. The waiting R4 queue is rebound to the new R8 coordinator PID.
