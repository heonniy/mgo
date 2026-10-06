# Headline execution, 2026-10-07

## Current checkpoint

Execution remains incomplete. Completed three-repeat attempts are not yet
stable enough for headline selection; preserve all samples without outlier removal.

| System | Local B16 / input256 | Local B64 / input512 |
| --- | --- | --- |
| OURS Near | Correctness PASS, timing UNSTABLE; confirmation queued | Preflight-only failure; new primary queued |
| DeepSpeed CPU offload | Correctness PASS, timing UNSTABLE; fixed-CPU confirmation queued | Correctness PASS, timing UNSTABLE; confirmation queued |
| MoE-Infinity repaired | Previous primary superseded by KV lifecycle fix; rerun queued | Previous primary superseded by KV lifecycle fix; rerun queued |
| llama.cpp static layers | Correctness PASS, timing UNSTABLE; confirmation queued | Corrected primary job running |

Live receipts are under `/home/hwlee/mgo-results/headline_r4_20261007`.
Current serial chain: `LLAMA_STATIC_RECOVERY_QUEUE.json` ->
`CLEANUP_RECOVERY_QUEUE.json` -> `LLAMA_SMALL_CONFIRMATION_QUEUE.json`.
The middle queue replaces five jobs that failed before model launch and their
skipped dependencies. Earlier queue receipts are historical, not active schedules.
See `queue_cleanup_recovery/README.md` for the preflight failures and bounded
release wait, `llama_measurement/README.md` for the required `--no-op-offload`
repair, and each baseline's archived audits for complete sample ranges.

The following setup narrative includes historical checkpoints; it must not be
read as the current completion status.

Owner authorizes completing all four baselines/systems, repairing execution
blockers rather than stopping on the first failure. OURS is LA_CA_NEAR.
Protocol source:7851e2ff. Scope remains only two R4/C30 cells, GPUs0,1,4,5,
global requests64/256, prompts256/512, exactly64 generated tokens (63 decode
intervals), three clean repeats after disjoint warmup, expert cache reset before every measured repeat (owner amendment).
No frozen routing or teacher forcing in the main table.

Frozen workloads are in HEADLINE_R4_WORKLOADS.json and headline_manifests/.
Selection is deterministic corpus order, first eligible prompt per conversation,
not the old policy-adversarial seed search. Warmup conversations are disjoint.

Budget accounting: include OURS physical P2 prefetch slots in the1843 global
expert-slot limit. MAIN capacities are [459,459,459,458], plus2 PREFETCH slots per
rank, yielding exactly1843 physical slots. Historical [461,461,461,460]+P2
characterization measurements are not reused as headline results. Full-pinned
CPU backing remains54GiB/rank,216GiB total. Dense weights/KV/workspace are
separately reported as total HBM. All baseline cache/streaming knobs must be
recorded against actual bytes, not percentage labels.

Initial implementation gaps: the characterization runtime reads frozen traces
and rejects native small-batch Gate history. A guarded live-routing path is being
implemented and validated; existing frozen paths stay available. Generation boundaries retain the loaded model/pinned CPU store and compiled
code, but measured repeats start with reset expert/controller/history state.

Dependencies are isolated under /home/hwlee/mgo-tools/headline-r4, keeping the
existing project environment intact. Initial upstream revisions:
- llama.cpp:3109914090564b4c5280f30896369d55b86bbdbf
- MoE-Infinity:9f819a6d43e043bded6e0692e5e58793e1623364
- DeepSpeed:0.19.7 (ZeRO-Inference CPU parameter offload, not FastGen).

Recovered setup failure: MoE-Infinity's required moe-store~=0.2.2 is unavailable
from the configured package index. Installed official EfficientMoE/moe-store
v0.2.2 (commit096f51f92ca9d698907d9a79120a7819ea094266) from source instead.
Raw setup logs are retained under the tools directory. No performance results
exist yet; build/setup is not a completed baseline.

Memory guards: initial host available >=384GiB for OURS; abort active worker
below96GiB. Never terminate unrelated GPU users or touch GPUs2,3,6,7. Stop owned
idle model loads only for GPU validation/timing and restore them after work.

## Owner cache-reset amendment

After warmup, reset expert cache before each measured batch/repeat. Keep model
loading, compiled kernels and full-pinned CPU backing intact. KV starts fresh for
every request batch. Measured prefill populates the expert cache and that state
continues into its decode; do not reset at the prefill/decode boundary.
Dynamic-cache systems must prove empty/reset residency before release. Static
llama.cpp GPU layer placement remains fixed: it is a configured weight partition,
not a request-adaptive expert cache. Record this distinction in the table.
This overrides warm-cache retention language in the original protocol/plan.
