# Physical CA stress validation

One bounded real-GPU validation of the strongest clean R8 stress cell.

Frozen cell:
- dataset: ShareGPT;
- R=8;
- local B=8 (global 64);
- cache=30% global;
- Gate eviction, W=128;
- substitution OFF;
- decode256;
- sample_seed=81;
- dp_seed=86;
- BR seed=42;
- policies: BR vs CA;
- environments: Env 1 and Env 2.

The selected workload is intentionally a communication-stress example, not a
dataset-average workload.

CPU resource expectation from commit 0b6da6e:
- CA peer bytes = 44,004,696,064;
- BR(seed42) peer bytes = 55,738,335,232;
- expected peer reduction = 21.0513%;
- across all eight predeclared BR seeds, reduction ranged only
  20.7803%--21.0513%, median 20.9679%.

Thus seed42 is the best observed BR seed, but the stress effect is not created
by one pathological random BR seed.
