# Critical-path admission follow-up

Status: prospective experiment plan. No new CPA policy or performance claim in this commit.

This packet follows the completed policy-regime result at `9c20847`.

The key observation is that FCA substantially reduced remote token-rank packets but increased forward/return NCCL residency and eventwise maximum expert GPU time. LA_CA preserved compute balance but produced only small, unsupported TPOT point gains.

The next question is therefore not how to reduce packet count further. It is:

> Which physical quantities determine multi-rank MoE layer completion time, and how much current-layer admission headroom remains if those quantities are optimized directly?

The experiment is staged:

1. isolate A2A split-shape sensitivity;
2. isolate return-collective arrival-skew sensitivity;
3. calibrate real expert execution cost tau(n);
4. build a microbench-calibrated critical-path model and validate it on the existing 9c20847 captures;
5. only if that model passes, solve an expensive current-layer placement oracle and physically replay it;
6. only if physical oracle headroom is meaningful, design an online CPA controller.

Do not implement another heuristic placement policy before stages 1-5 answer these questions.
