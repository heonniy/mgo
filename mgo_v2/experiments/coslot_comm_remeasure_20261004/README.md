# CoSLoT communication remeasurement (2026-10-04)

This packet isolates the communication implementation from the existing
BR/CA placement policy.

## What is held fixed

- The already validated frozen PLAN is reused.  Do **not** create a new PLAN.
- Expert placement, substitution, eviction, cache actions, H2D fetches, model,
  prompts, and generated-token correctness are unchanged.
- The same COMPILE -> MEASURE -> COUNTERS protocol and thermal/memory guards
  from `env_e2e_tpot_offload_20261003` are reused.

## What changes

`--comm-mode current` keeps the existing transport:

1. token->destination-rank hidden-state A2A,
2. token->destination-rank weight A2A,
3. expert-output return A2A.

`--comm-mode coslot` expands each frozen token->rank packet into effective
`(token, expert)` route rows and uses the fast CoSLoT transport shape:

1. byte-pack `hidden | expert_id | gate_weight` and issue **one** uneven
   `all_to_all_single`,
2. execute the same frozen experts,
3. issue **one** uneven return `all_to_all_single`.

So the CoSLoT mode intentionally trades more activation bytes (duplicate hidden
states when several experts on one rank serve the same token) for fewer NCCL
collective launches.

This is a transport-faithful CoSLoT remeasurement, not a rerun of the entire
legacy controller.  Global-demand/controller collectives remain excluded from
MEASURE because the purpose is to isolate transport under the same frozen
policy decisions.

## Correctness gates

Every COMPILE/MEASURE run must match the original PLAN's:

- `token_hash`,
- `state_hash`,
- route/action hashes,
- cache capacity and expert actions.

The CoSLoT fused payload also carries expert IDs.  Outside MEASURE, received
expert IDs are checked against the frozen execution groups.

## Counters

COUNTERS additionally reports:

- `actual_peer_bytes`: activation/output bytes, compatible with the previous
  counter,
- `actual_wire_bytes`: all NCCL payload bytes including route metadata,
- `actual_collective_calls`: number of A2A calls issued by one rank.

For 257 forward steps x 48 MoE layers = 12,336 layer events, the expected call
counts are:

- current: 37,008 A2A calls/rank/generation,
- CoSLoT: 24,672 A2A calls/rank/generation.

## Primary rerun

Start with the same R cell and BR vs CA:

```bash
cd mgo_v2
PYTHONPATH="$PWD:$PWD/scripts" \
python scripts/run_coslot_comm_remeasure.py \
  --cell R --environment env1 --policies BR CA --repeats 3
```

If that is stable, repeat with `--environment env2`.  The output directories
are suffixed with `_coslot`, so the prior measurements are not overwritten.

Interpretation should compare both:

1. BR vs CA **within CoSLoT mode**: does placement help under the old
   expert-route transport?
2. current vs CoSLoT for the same policy: is the observed TPOT dominated by
   NCCL launch count or by transferred bytes?
