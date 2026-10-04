# R8 fixed-route physical decode64 replay

Exploratory optimized stress screen, not dataset-average or autoregressive
quality evidence. Fetch/H2D tolerance is0.1% relative to BR, not exact equality.
Single-shot observations are screening evidence; no confidence intervals.
Confirmation is exactly one further BR+CA pair only for >=1% initial CA gain.

| Workload | Comparison | Observation | E2E gain | Decode-wall gain | TPOT gain |
|---|---|---:|---:|---:|---:|
| R8_s484_d13_b73 | BR_to_CA_A3 | 1 | -0.10% | -0.11% | -0.11% |
| R8_s484_d13_b73 | BR_to_CA_A2 | 1 | -0.42% | -0.44% | -0.44% |
| R8_s484_d13_b73 | A3_to_A2_BR | 1 | 2.63% | 2.65% | 2.65% |
| R8_s484_d13_b73 | A3_to_A2_CA | 1 | 2.32% | 2.33% | 2.33% |

Raw seconds are in the timing CSV, with all mechanism counters alongside.
Every physical fetch/peer/Critical_total reproduces its selected CPU schedule.
A2 removes routing-weight traffic:9360→6240 A2A calls/rank over prefill+64
decode, and9216→6144 for decode alone. No full schedule warmup precedes each
measurement. Separate untimed COUNTERS passes validate A2 and compiler coverage.
Full64 teacher-forced argmax hashes agree A3/A2; the first decode forward
compares weighted partials across all48 layers. No further sweep follows.
