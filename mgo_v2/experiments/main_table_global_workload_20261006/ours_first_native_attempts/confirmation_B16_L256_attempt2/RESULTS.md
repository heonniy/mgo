# OURS small-cell bounded confirmation

All3 repeats pass correctness, manifest/raw clocks, cold expert state, physical1843 slots and no compilation in timing. Timing remains UNSTABLE. The bounded confirmation is complete; do not discard repeat1 or add an automatic repeat loop.

- TTFT: median 4.041778560s; range [3.992524478, 5.816702441]s; spread 39.510%.
- TPOT: median 0.769797787s; range [0.758135499, 0.787127641]s; spread 3.757%.
- E2E: median 52.539039150s; range [51.755060927, 55.405743818]s; spread 6.858%.

Both independent launches reproduce a longer first primary TTFT (~5.8s) followed by ~4.0s. This is evidence of a repeat-position effect, not proof of its cause. Cold expert receipts and no-compile guards pass every repeat. Raw sample1 remains part of both reported triplets. No stable headline selection is made.
