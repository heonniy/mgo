# M11 exact-return amendment

The owner's latest instruction authorizes repairing out-of-plan failures and
continuing toward the best physically validated LA gain. This amendment keeps
correctness as a hard gate rather than silently weakening exact output parity.

The requested destination-rank BF16 partial aggregation changes addition
association. It cannot generally reproduce sequential source expert-order BF16
rounding from one hidden-sized partial. The physical eight-rank B128/B256 tests
exercise random, uneven and zero-peer traffic. In Env1, rank-partial compression
changed 5,098,605 output elements (maximum absolute difference 0.25), while the
exact fused path changed zero. These are communication test tensors, not model
accuracy measurements. No model gain is inferred from them.

Use the following common communicator for the three final arms:

- One fused forward payload A2A per layer, coalesced by token/destination rank.
  Each packet contains the hidden vector, expert IDs and original BF16 weights.
- One return payload A2A per layer containing each weighted expert contribution,
  preserving the old source expert-order accumulation exactly.
- No weight/count payload A2A and no production plan broadcast.
- Rank-partial return compression remains a diagnostic negative control and is
  excluded from primary timing/default selection.

This is an explicit deviation from the planned hidden-sized destination partial:
return traffic is not compressed to one vector per token/rank. It retains exact
two-round communication and all other prefetch/overlap improvements. All final
arms use this same communicator and wire format, so their comparison stays fair.
M11 is recorded as `PASS_EXACT_TWO_A2A / FAIL_RANK_PARTIAL_BITWISE`, not as full
compliance with the original return-compression requirement. Do not claim a
return-bandwidth reduction that this implementation does not provide.

Env1/Env2 physical receipts retain every tested case, both modes, collective
counts, wire bytes and numerical differences. Full-model parity is additionally
required at M12 before any tuning or primary timing.
