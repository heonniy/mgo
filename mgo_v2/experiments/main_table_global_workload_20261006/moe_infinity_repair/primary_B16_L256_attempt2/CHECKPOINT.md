# Partial primary checkpoint

Attempt2 has completed warmup and primary repeat1; repeats2/3 are pending.
Raw authoritative job: /home/hwlee/mgo-results/headline_r4_20261007/infinity_B16_L256_primary2

Repeat1: TTFT5.197430501s, TPOT3.063612890s, E2E198.205042574s.
All64 requests generated64 tokens. Initial expert residency0; peak charged bytes17392730112 equals the cap. EAM3072 calls,4729595 candidate submissions,325502 priority evictions,82145 lower-priority prefetch rejections. Candidate counts are submissions across layers/steps, not distinct experts or prefetch hits. No final median, stability or cross-system gain claim yet.

Repeat2 also PASS: TTFT4.883946322s, TPOT3.070051901s, E2E198.297216084s. Repeat3 running. Pair TTFT difference6.22% is unstable despite TPOT0.21% and E2E0.05%. Native greedy outputs differ in325/4096 token positions across12/64 requests (earliest difference at decode token index6). Finite-output checks passed; do not claim bitwise determinism or infer the cause from timings alone. See PARTIAL_AUDIT.json.
