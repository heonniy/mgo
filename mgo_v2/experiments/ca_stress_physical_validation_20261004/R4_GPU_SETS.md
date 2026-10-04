> **Repetition override:** `REPETITION_AMENDMENT.md` supersedes all earlier three-repeat / two-extra-repeat instructions below. Use two clean samples, at most one conditional third, independently per environment/policy.

# Owner amendment: two fixed physical GPU sets

The owner's message following a121c85 authorizes the same frozen R4 study twice:

| Variant | Logical rank 0 | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|---:|
| R4_0123 | GPU0 | GPU1 | GPU2 | GPU3 |
| R4_0146 | GPU0 | GPU1 | GPU4 | GPU6 |

This expands only the physical GPU-set axis of R4_AMENDMENT.md. Run R4_0146,
then R4_0123, after the existing R8 study fully exits and commits. Do not modify
or interrupt the running R8 driver, workers or matrix. A failed R8 execution
requires attention; do not treat an infrastructure failure as completion.

Both R4 variants preserve sample205/DP4, BR73, exact request/rank membership,
local B8, cache30 (1843 total slots), Gate W128, substitution OFF and decode256.
The physical PLAN policy explicitly initializes the NumPy/Numba random stream
with seed73. CA is deterministic. The existing R8 default seed42 is untouched.

Logical rank r uses CPUs [24*r, 24*r+23] in both variants. Holding CPU affinity
constant avoids changing a second axis with GPU membership. These guest CPU
sets are not interpreted as host NUMA domains. The selected physical GPU and
logical rank are recorded in every model receipt and both transport preflights.

Each variant has separate PLAN, compile cache, COMPILE, MEASURE and COUNTERS
artifacts. Use the original counterbalanced 12-run order for each variant.
The inherited 5% spread gate adds exactly two repeats per policy/environment
only when triggered (at most 20 measured generations per variant). No other
seed/request/cache/rank sweep is authorized. CPU validation runs only after R8
finishes, avoiding interference with active R8 timing. Validate the entire
selected 256-step incremental policy against the frozen CPU reference before
GPU phases, then validate physical PLAN hashes across warmup/timing/counters.

Stop all project-owned resident model workers during scientific GPU phases,
including workers on GPUs outside the selected four, to avoid host contention.
Do not stop foreign work. Enforce existing memory/temperature guards. Keep the
other GPUs idle through the two R4 variants; restore the owner's real-model
load on all eight after both finish or a failure is checkpointed. The R8
finalizer retains its existing restoration behavior; the R4 driver safely
stops those owned workers before starting, without changing R8 behavior.

Results are stored in R4_0123/ and R4_0146/ under this packet; large artifacts
are in the matching subdirectories of the existing raw-results root. Commit
and push each phase. R4_RESULTS.md and R4_R8_comparison.csv will compare the
two sets and the R8 best case. R8 has a different selected workload and global
batch, so an R4/R8 contrast cannot isolate the causal effect of rank count.
All rows remain optimized communication-stress examples, not dataset averages.
