# Physical local/remote TPOT and E2E study — 2026-10-01

This experiment measures two things in the validated mgo_v2 runtime:

1. how actual MoE/NCCL latency changes as expert service becomes more local or more remote;
2. how much the final communication-aware admission policy changes TPOT and end-to-end generation latency under real CPU expert offloading.

The runtime is already validated for R1/R4/R8 native parity, physical expert H2D accounting, direct resident-slot execution, substitution, Coverage eviction and all admission policies. This folder is the canonical experiment contract for the next measurement stage.

Do not retune policy coefficients after timing results are visible.
