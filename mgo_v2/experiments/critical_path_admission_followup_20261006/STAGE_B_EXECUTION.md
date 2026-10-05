# Stage B execution — validate a physical critical-path model

Parent checkpoint: `dd28e4d` (A1-A3 complete)

This stage is **offline analysis only**. Do not run the oracle or a new online policy yet.

## Goal

Test whether the three measured mechanisms from Checkpoint A are sufficient to explain the earlier policy-regime result:

```
LA_CA ~= BR < OLD_CA < FCA
```

especially the fact that FCA reduces packet count but becomes slower.

The model must be calibrated from A1-A3 only. The old BR/OLD_CA/FCA/LA_CA timings are validation data, not fitting data.

## 1. Event inputs

For every decode MoE event in the existing C30/C60 captures, reconstruct:

- directed token-rank split matrix `S[src,dst]`;
- packet count and bytes;
- per-rank list of executing experts;
- row count `n_e` for every expert;
- current owner rank for every expert;
- observed forward NCCL duration;
- observed expert GPU duration;
- observed return NCCL duration.

No future route is read.

## 2. Forward transport model

Use A1 to construct a small lookup/interpolation model:

```
T_fwd_hat = F(packet_bytes, total_packets,
              max_send, max_recv, max_incident, max_peer_edge)
```

Start with the smallest model that explains the A1 synthetic curves.

Do not fit coefficients to full-model captures.

A1 established that rank concentration matters, but real BR/FCA trace matrices are near the balanced regime. The model must preserve that distinction.

## 3. Expert service / rank-ready time

Use the A3 `tau(n)` table.

For each rank:

```
T_expert_hat[r] = sum_{e owned by r} tau(n_e)
```

Then define relative rank ready time after forward delivery:

```
ready[r] = T_expert_hat[r]
ready_max = max_r ready[r]
```

Also retain the old row-count estimate for comparison:

```
rows[r] = sum_e n_e
```

This lets us test whether calibrated GPU milliseconds explain the data better than raw rows.

## 4. Return collective model — avoid double counting

Checkpoint A2 showed that a late rank's arrival delay transfers to collective completion with slope ~0.999.

Therefore do **not** add "expert max" and the full observed return residency as two independent costs.

Instead use:

```
T_ret_wire_hat = R(S)
T_layer_hat = T_fwd_hat + ready_max + T_ret_wire_hat
```

and, for rank-local return residency diagnostics:

```
T_ret_residency_hat[r]
  = T_ret_wire_hat + (ready_max - ready[r])
```

Interpretation:
- the last rank pays mainly wire/base collective time;
- earlier ranks additionally wait for the late rank;
- that waiting is already caused by expert completion skew.

This is the main Stage-B hypothesis.

## 5. H2D handling

Do not silently fold H2D into communication or expert time.

For Stage B:
- validate the communication+expert mechanism first;
- report observed H2D separately;
- because mandatory fetch quotas are balanced, do not use H2D to fit the placement model;
- flag events/policies where future cache trajectory changes H2D materially.

If the model ordering only works after fitting an H2D coefficient to old policy timings, Stage B fails.

## 6. Required comparisons

For each C30/C60 and BR/OLD_CA/FCA/LA_CA, report:

1. observed vs predicted forward duration;
2. observed vs predicted eventwise max expert duration;
3. observed vs predicted return-residency distribution;
4. observed vs predicted combined critical-path score;
5. row-only model vs tau(n) model.

Primary validation gates from PLAN.md remain:

- qualitative ordering: `LA_CA ~= BR < OLD_CA < FCA`;
- event-level Spearman >= 0.60;
- median absolute relative error <= 25% for the corresponding profiled phase aggregate.

Also report these mechanism-specific checks:

```
corr(predicted expert max, observed expert max)
corr(predicted return residency, observed return residency)
corr(predicted layer score, observed layer critical timing)
```

## 7. Decision

### PASS

Proceed to Stage C only if:
- the model reproduces the policy ordering without fitting to policy TPOT;
- at least one of the physical models meets both numerical gates;
- tau(n) is not worse than raw rows for expert-time prediction;
- predicted arrival-wait behavior is consistent with observed return-residency inflation.

### FAIL

Stop before oracle if:
- packet/split + tau(n) + arrival skew cannot explain the FCA slowdown;
- the model requires coefficients fitted to the old TPOT results;
- event-level correlation remains weak.

A Stage-B failure means another missing physical mechanism exists. Do not compensate by inventing a weighted CA/LA heuristic.

## 8. Outputs

Create:

- `CRITICAL_PATH_MODEL.json`
- `MODEL_EVENT_FEATURES.csv/json`
- `MODEL_VALIDATION.csv/json`
- `STAGE_B_RESULTS.md`
- source/provenance hashes.

Do not run new GPU inference for Stage B unless an existing artifact needed for event reconstruction is genuinely missing. If that happens, stop and report the missing evidence rather than silently collecting a different workload.

## 9. Stop

After publishing Stage-B validation, stop for owner review.

Do **not** run Stage C oracle automatically.
