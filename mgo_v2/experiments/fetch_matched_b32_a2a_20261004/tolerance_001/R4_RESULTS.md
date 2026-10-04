# R4 fixed-route physical decode64 replay

Exploratory optimized stress screen, not dataset-average or autoregressive
quality evidence. Fetch/H2D tolerance is0.1% relative to BR, not exact equality.
Single-shot observations are screening evidence; no confidence intervals.
Confirmation is exactly one further BR+CA pair only for >=1% initial CA gain.

| Workload | Comparison | Observation | E2E gain | Decode-wall gain | TPOT gain |
|---|---|---:|---:|---:|---:|
| R4_s251_d42_b99 | BR_to_CA_A3 | 1 | 0.23% | 0.25% | 0.25% |
| R4_s251_d42_b99 | BR_to_CA_A2 | 1 | 0.78% | 0.79% | 0.79% |
| R4_s251_d42_b99 | A3_to_A2_BR | 1 | 1.54% | 1.58% | 1.58% |
| R4_s251_d42_b99 | A3_to_A2_CA | 1 | 2.08% | 2.11% | 2.11% |
| R4_s415_d8_b42 | BR_to_CA_A3 | 1 | 2.39% | 2.46% | 2.46% |
| R4_s415_d8_b42 | BR_to_CA_A3 | 2 | 0.19% | 0.19% | 0.19% |
| R4_s415_d8_b42 | BR_to_CA_A2 | 1 | 0.60% | 0.59% | 0.59% |
| R4_s415_d8_b42 | A3_to_A2_BR | 1 | 4.66% | 4.75% | 4.75% |
| R4_s415_d8_b42 | A3_to_A2_CA | 1 | 2.91% | 2.92% | 2.92% |

Raw seconds are in the timing CSV, with all mechanism counters alongside.
Every physical fetch/peer/Critical_total reproduces its selected CPU schedule.
A2 removes routing-weight traffic:9360→6240 A2A calls/rank over prefill+64
decode, and9216→6144 for decode alone. No full schedule warmup precedes each
measurement. Separate untimed COUNTERS passes validate A2 and compiler coverage.
Full64 teacher-forced argmax hashes agree A3/A2; the first decode forward
compares weighted partials across all48 layers. No further sweep follows.
