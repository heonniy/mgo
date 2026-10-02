# EXACT-ONLY PAYLOAD CAPTURE — minimal unblock for crossover

Status: authorized follow-up after `0c09fae`.

## Why a new short capture is needed

Existing B8 traces were collected with substitution enabled. Raw router choices and token-origin ranks are present, but a substituted-away raw expert has no observed exact execution owner. Therefore an exact rank-pair payload distribution cannot be reconstructed without inventing a counterfactual placement.

Resolve this with **one short exact-only physical capture**, not a new performance study.

## Fixed capture

Use exactly:

- physical GPUs: **0,1,4,5**;
- world size: 4;
- Qwen3-30B-A3B-Instruct-2507 BF16;
- local batch B8 / global batch 32;
- cache ratio 30%;
- **substitution disabled**;
- **replication disabled**;
- **LRU eviction**;
- admission: existing deterministic **Balanced Random (P0, seed 42)**;
- one prefill + **8 decode forwards**;
- normal T0 NVSwitch transport;
- one run only.

This run is for payload characterization only. Do not use its wall time as a performance result.

## Capture the actual execution traffic

Do not infer destination ranks afterward if the runtime can expose the real counts.

For every layer event save:

- raw selected experts;
- token origin ranks;
- `owner_by_expert` after exact misses are admitted;
- dispatch send counts by ordered rank pair;
- combine/return send counts by ordered rank pair;
- effective token routes;
- cache/plan hash.

With substitution disabled, every raw selected expert required by the exact route must have an execution owner before dispatch.

Fail the run if any selected exact expert lacks an owner.

## Payload accounting

Qwen3 hidden size is 2048 and BF16 is 2 bytes per element.

For each non-self ordered rank pair and each decode layer event, compute actual nonzero bytes from the recorded runtime send counts.

Keep dispatch and combine distributions separate.

Report:

- nonzero message count;
- p50;
- p90;
- p99;
- maximum;
- bucket fractions:
  - <=64 KiB
  - 64–256 KiB
  - 256 KiB–1 MiB
  - 1–4 MiB
  - >4 MiB.

Also report the union distribution of dispatch+combine messages as a convenience, but do not replace the separate phase distributions.

## Five-size crossover immediately after capture

After the exact-only payload file passes validation, run only:

```text
32 KiB
64 KiB
256 KiB
1 MiB
4 MiB
```

per peer, using the same `all_to_all_single` API.

Compare:

### T0

Normal NVSwitch / direct P2P.

### R3

Validated non-P2P SHM condition:

```bash
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
export NCCL_IB_DISABLE=1
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Expected path: `SHM/direct/direct`.

For each size and mode:

- 10 warmups;
- 30 timed iterations;
- max-rank median and p90;
- effective bandwidth.

Run two lightweight counter-ordered passes only:

- pass 0: T0 -> R3;
- pass 1: R3 -> T0.

INFO logging only for the first size of each mode to confirm transport, then disable it.

## Decision

For each size compute:

[
ratio(S)=T_{R3}(S)/T_{T0}(S)
]

Continue the H100 synthetic branch only if R3 becomes **>=1.5x slower** at a size that lies at or below the observed exact-only p90/p99 payload range.

Otherwise stop synthetic H100 P2P-off work and use the real no-NVLink server for the communication-sensitive arm.

Do not change NCCL channels/protocols, batch messages, search more transport knobs, implement replication, or launch Stage 1 during this follow-up.

## Bounded outputs

Commit:

- `exact_payload_capture_summary.json`
- `payload_distribution.csv/json`
- `payload_crossover.csv`
- `PAYLOAD_CROSSOVER_RESULTS.md`
- one compact T0 transport log and one compact R3 transport log.

Stop for owner review immediately after this packet.
