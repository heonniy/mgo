# DeepSeek ZeRO-3 expert collective repair

The stock DeepSeek-V2 MoE loop skips locally empty experts. In ZeRO-3 each
expert call gathers its partitioned parameters across all ranks, so different
local expert selections produce different collective call sequences. During
`mt2_deepseek_sharegpt_b16_l512_deepspeed_smoke_v5`, three ranks remained in
routed layer 1 while the fourth had entered layer 2. The two-token calibration
could not complete.

`deepseek_moe_loop.py` now all-reduces the 64 expert token counts once per
routed layer. Every rank calls the resulting global expert union in the same
order, passing zero-row input when an expert has no local tokens. It retains
the original local output order and combine arithmetic. A two-rank CPU check
with disjoint expert selections produced exactly the stock local output and
the same global call order on both ranks.

The DeepSeek calibration also enforces its all-parameter C30 byte bound at
every native fetch using DeepSpeed's O(1) live-parameter counter, with a full
parameter scan every 64 fetches. The separately reported expert peak is a
periodic sample, while the all-parameter peak is the exact enforced bound.
The Qwen path retains its prior full-scan calibration.

The repaired four-GPU DeepSeek smoke
`mt2_deepseek_sharegpt_b16_l512_deepspeed_smoke_v6` passed its two-token
warmup and target. The four ranks recorded 10,637 fetch-boundary checks, 167
full counter cross-checks, an all-parameter peak of 1,494,772,736 bytes
against a 2,158,362,624-byte per-rank budget, and 9.2–9.9 seconds for the
two-token calibration. This smoke establishes execution and budget validity;
the full 64-token three-repeat row is measured separately.
