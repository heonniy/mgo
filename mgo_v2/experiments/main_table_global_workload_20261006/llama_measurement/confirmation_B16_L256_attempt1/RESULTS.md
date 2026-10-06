# llama.cpp small-cell bounded confirmation

PASS: all three repeats satisfy frozen request/token/raw-clock, corrected static expert placement/budget and <=5% TTFT/TPOT/E2E spread gates. Select this complete confirmation triplet. The earlier unstable triplet remains archived; no further repeats.

- TTFT: median 37.702121201s; range [37.441671241, 38.006756360]s; spread 1.498%.
- TPOT: median 0.257712683s; range [0.257712619, 0.261402397]s; spread 1.425%.
- E2E: median 54.170472201s; range [53.677566241, 54.242655360]s; spread 1.046%.

BF16 weights, native FP16 KV.14 GPU expert layers/34 CPU expert layers with host-op offload disabled; GPU KV smoke allocation evidence remains in static_smoke1. Per-primary full allocation telemetry is not claimed.
