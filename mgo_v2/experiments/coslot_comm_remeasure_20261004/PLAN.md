# PLAN

## Question

Does the small BR->CA TPOT gain persist when the exact same frozen placement
schedule is executed with the CoSLoT fast-A2A communication structure?

## Controlled variable

Only the communication representation changes.

- current: coalesced token->rank packets, 3 A2A calls/layer.
- coslot: effective token-expert routes, fused forward payload, 2 A2A
  calls/layer.

No new policy planning is allowed for the CoSLoT run.

## Primary matrix

1. R / env1 / BR
2. R / env1 / CA
3. If stable, R / env2 / BR
4. If stable, R / env2 / CA

Use one COMPILE, three MEASURE repeats, and one COUNTERS run per cell/policy.

## Pass/fail gates

- Original PLAN token/state hashes must match.
- No compile in MEASURE.
- No schedule regeneration.
- Same H2D/cache actions for matching policy.
- COUNTERS must show 24,672 A2A calls/rank for CoSLoT over one full generation.
- Timing is only interpreted if the existing thermal/noise guards pass.

## Main outputs

Report median TPOT and E2E for BR and CA, plus:

- CA vs BR TPOT delta,
- activation peer bytes,
- total wire bytes,
- A2A call count.

A larger CA gain under CoSLoT would support the hypothesis that transport
representation changes the visibility of placement benefits.  A similarly
small gain would indicate that the bottleneck is elsewhere rather than merely
the current token->rank coalescing path.
