# Packet-aware online admission policy candidate

Status: code-only policy checkpoint; no performance claim.

## Motivation

The refactored fused transport communicates one packet per token/destination
rank, not one packet per expert route. If several experts selected by the same
token are placed on the same remote rank, they share one forward hidden packet
and one coalesced return packet. A static expert-locality cost therefore misses
the co-location effect that matters to the physical transport.

The new policy keeps the existing balanced mandatory-H2D quota unchanged and
changes only which missing expert is assigned to each quota slot.

## FCA: fanout-aware communication admission

For every token, keep the set of destination ranks already required by resident
experts and previously committed misses. For a candidate placement e -> r:

- local origin==r contributes no remote packet;
- if r is already a destination for that token, the marginal packet cost is 0;
- otherwise it adds one remote token/rank pair and increments the incident load
  of both the source rank and destination rank.

Greedy objective:
1. minimize the maximum rank-incident remote packet count;
2. minimize newly-created remote token/rank pairs;
3. deterministic rank-id tie break.

After every committed miss, update the token destination masks before scoring
the next miss. This captures packet coalescing that the old static Hungarian CA
cannot represent.

## LA+CA joint candidate

The first joint policy is deliberately lexicographic so no arbitrary lambda
mixes expert rows and communication packets:

1. minimize predicted critical-rank expert load;
2. among equal-compute candidates, minimize rank-incident packet makespan;
3. minimize new remote packet count;
4. lower destination load;
5. larger local demand;
6. deterministic rank id.

This preserves the current LA objective while exploiting communication
coalescing whenever it does not worsen predicted compute makespan.

A later hardware-calibrated policy may replace this proxy with
T_fwd_A2A + max_r(T_expert,r) + T_return_A2A in measured milliseconds after
B/cache characterization establishes a stable communication cost model.

## Physical adapter policy IDs

- 0: BR
- 1: existing demand-locality CA
- 3: historical static fanout Hungarian CA
- 4: LA
- 5: FCA, dynamic token/rank packet-aware CA
- 6: LA_CA, lexicographic joint candidate

The refactored controller exposes FCA and LA_CA as opt-in names. Existing CA
semantics and archived proofs are unchanged.

For predicted next-layer prefetch, the current predictor provides expert/rank
demand but not token-level co-routing. FCA therefore falls back to demand
locality, and LA_CA falls back to LA for speculative placement. Packet-aware
prefetch is deferred until the predictor exposes token-level destination
co-occurrence.
