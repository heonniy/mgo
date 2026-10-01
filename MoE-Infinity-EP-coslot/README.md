# MoE-Infinity-EP

Multi-process EP + DP rewrite of [MoE-Infinity](../MoE-Infinity) with NCCL-based inter-process communication, a global cache view, and pluggable fetch-dispatch / placement policies.

This repo does **not** fork MoE-Infinity. It depends on it as an editable install and reuses the model loaders, the C++ `ArcherPrefetchHandle` (PCIe fetch), and the per-rank `ExpertDispatcher` (as a single-GPU local executor). The multi-GPU orchestration and the offload engine are replaced.

## Status

- **Milestone 1 (in progress)**: distributed scaffolding + weight load.

See [`/home/work/.claude/plans/replicated-jingling-umbrella.md`](../../../.claude/plans/replicated-jingling-umbrella.md) for the full plan.

## Quick start (Milestone 1 smoke)

```bash
# One-time: install upstream MoE-Infinity (editable) and this package
pip install -e ../MoE-Infinity
pip install -e .

# Multi-process boot test (uses all visible GPUs by default)
bash scripts/run_smoke.sh configs/qwen3_235b_auto.yaml
```

## Layout

```
moe_infinity_ep/
├── launch/        # torchrun env, NCCL group construction, MoE_EP entry
├── runtime/       # DistributedOffloadEngine, EPExpertExecutor
├── cache/         # GlobalCacheView + sync protocol
├── policies/      # FetchDispatchPolicy / PlacementPolicy ABCs + naive defaults
├── exec/          # per-layer choreography, NVLink router, PCIe fetcher
├── instrument/    # always-on counters and trace dump
├── models/        # model block shims (Qwen3MoEBlockEP)
└── utils/         # config loader
```
