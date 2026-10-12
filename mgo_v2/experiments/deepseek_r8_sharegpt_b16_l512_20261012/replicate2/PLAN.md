# DeepSeek R8 serial + inline H2D: replicate run with an added FAST+Worst arm

Same setup as `../PLAN.md` (serial, inline H2D, prefetch OFF, BR prefill, same frozen NEAR teacher
tokens `../FROZEN_TOKENS.json`). Fresh session; no rows reused from the first run.

Arms: NEAR, FAST(+Near), FAST+Random, FAST+Worst (`FAST_WORST`, mirror of Near: max compute
imbalance, then min locality), and RANDOM_QUOTA(+Near).
One smoke for FAST_WORST comes first. Then three rounds with rotated order, 2 target repeats per job
(6 samples per arm). NEAR is the baseline.
