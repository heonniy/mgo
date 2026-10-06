# MoE-Infinity small-cell bounded confirmation

All three repeats pass frozen request/raw-clock, output shape, cold expert residency, EAM, priority eviction, exact per-GPU/global byte caps and actual KV release checks. TTFT remains UNSTABLE; TPOT/E2E individually pass5% spread. No outlier removal, stable whole-row selection or additional automatic repeats.

- TTFT: median 5.020633884s; range [4.839623998, 5.136096500]s; spread 5.931%.
- TPOT: median 3.073081183s; range [3.067091553, 3.112734072]s; spread 1.480%.
- E2E: median 198.740211023s; range [198.066391861, 201.122880434]s; spread 1.534%.

Every primary restores the same warmup EAM history and starts with zero expert residency. The repaired baseline retains its validated source/binary fingerprints; global expert cap17392730112bytes is respected throughout. KV weakrefs release after every batch.
