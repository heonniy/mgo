# Completed primary attempt2; headline stability not passed

Raw authoritative job: /home/hwlee/mgo-results/headline_r4_20261007/infinity_B16_L256_primary2

All3 repeats pass manifest/count, finite logits, EAM, priority eviction, empty cache start and per-GPU expert budget checks. Median TTFT4.883946322s (range4.868799704–5.197430501), TPOT3.063612890s (3.036429477–3.070051901), E2E198.205042574s (196.163856760–198.297216084).

TTFT spread6.5945% exceeds the5% headline gate; TPOT1.1000%, E2E1.0799%. Preserve all3 samples; this is not a stable headline row and no cross-system gain is claimed. FINAL_ATTEMPT_AUDIT.json is current; PARTIAL_AUDIT.json preserves the earlier two-sample checkpoint.

Native greedy outputs are not bitwise identical across repeats (see partial audit). Candidate counts are submissions across layers/steps, not distinct experts or prefetch hits. HBM is a1Hz sampled device peak; host RSS is process RSS and not uniquely owned physical RAM.
