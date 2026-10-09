# Native ZeRO-3 Qwen3 MoE leaf adaptation

Transformers 4.57.1 calls only the experts selected by a rank's local tokens.
Different ranks can therefore call different expert modules in different orders.
DeepSpeed 0.17.6's native ZeRO-3 parameter collection must encounter a consistent
collective sequence. Its `set_z3_leaf_modules` API supports dynamic MoE execution
by collecting all child parameters at the parent block before conditional calls.

The PCIe Qwen3 worker now marks all 48 `Qwen3MoeSparseMoeBlock` modules before
engine initialization, using that native API. Routing, expert arithmetic, weights
and generated-token semantics are unchanged. Each parent fetch includes all 128
experts and the router: 1,208,483,840 BF16 bytes per block. This fits the
4,348,182,528-byte per-rank parameter budget individually; simultaneous residency
and prefetch still require the existing GPU calibration and live budget guards.
Fetching inactive experts is a documented consequence of generic ZeRO offload
and differs from OURS' demand expert cache. It is not FastGen.

The adaptation is scoped to `MGO_PCIE_HOST=1` and Qwen3. Legacy and DeepSeek
worker paths keep their previous setup. A per-rank `zero_leaf_rank*.json`
receipt records marked blocks, full parameter sizes and the bound. The launcher
records the helper source digest separately from the worker digest.

`DEEPSPEED_LEAF_PREFLIGHT.json` proves full-size setup on a 48-block BF16 meta
skeleton and exact CPU output/router-logit parity before/after native leaf
marking on a small block. No CUDA context or physical GPU execution is claimed.
The forthcoming four-rank calibration, full warmup and bounded primary runs
must establish collective progress, finite output, C30 residency and resource
correctness before any baseline performance claim.

Reproduce from repository root:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data2/esjung/envs/mgo-pcie/bin/python mgo_v2/scripts/pcie_deepspeed_leaf_preflight.py \
  --out /tmp/DEEPSPEED_LEAF_PREFLIGHT.json
```

The preflight receipt hashes the local installed DeepSpeed leaf API,
Transformers conditional-expert implementation, frozen model configuration and
setup/preflight scripts as the primary implementation evidence.
