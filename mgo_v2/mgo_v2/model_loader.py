"""Qwen3 dense GPU weights plus the controller-owned C++ expert store.

No legacy model loader, RPC executor, prefetcher or placement manager is
constructed. The native extension is used only for host backing and slots.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from .native import load_slot_extension

EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
PROJECTIONS = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}


def checkpoint_identity(model_path):
    root = Path(model_path).resolve()
    index = root / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    return {"model": str(root), "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
            "shards": {name: {"bytes": (root / name).stat().st_size,
                                "mtime_ns": (root / name).stat().st_mtime_ns}
                       for name in sorted(set(weight_map.values()))}}, weight_map


def write_expert_store(destination, weights, identity):
    """Write the legacy x86-64 tensor index once, with aligned BF16 payloads.

    Unlike offload() per tensor, this does not rewrite an ever-growing index
    thousands of times. A manifest is committed last; incomplete stores must
    never be opened by model workers.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty store: {root}")
    records = []
    offset = 0
    with (root / "archer_param_0").open("wb") as data:
        for tensor_id, tensor in weights:
            tensor = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            nbytes = tensor.numel() * 2
            data.write(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
            padding = (-nbytes) % 4096
            if padding:
                data.write(bytes(padding))
            shape = tuple(tensor.shape)
            # See core/aio/archer_tensor_index.cpp: key, file_id, offset,
            # size_t size, int64 ndim, dimensions; then TensorOptions
            # (pinned, requires_grad, ScalarType::BFloat16, CPU index/type, strided).
            record = struct.pack("<IIqQq", tensor_id, 0, offset, nbytes, len(shape))
            record += struct.pack("<" + "q" * len(shape), *shape)
            record += struct.pack("<??bbbb", False, False, 15, -1, 0, 0)
            records.append(record)
            offset += nbytes + padding
    with (root / "archer_index").open("wb") as index:
        index.write(struct.pack("<I", len(records)))
        for record in records:
            index.write(record)
    manifest = {"format": "archer-x86_64-bf16-v1", "identity": identity,
                "tensors": len(records), "bytes": offset,
                "index_sha256": hashlib.sha256((root / "archer_index").read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_checkpoint_store(model_path, destination):
    identity, weight_map = checkpoint_identity(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen3_moe":
        raise ValueError("only Qwen3 MoE is supported by this loader")
    expected = {f"model.layers.{l}.mlp.experts.{e}.{p}.weight"
                for l in range(config.num_hidden_layers) for e in range(config.num_experts)
                for p in PROJECTIONS}
    actual = {name for name in weight_map if EXPERT.fullmatch(name)}
    if actual != expected:
        raise ValueError("checkpoint does not contain the expected Qwen3 experts")
    root = Path(destination)
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("offload manifest belongs to a different or modified checkpoint")
        if manifest.get("format") != "archer-x86_64-bf16-v1" or manifest.get("tensors") != len(expected):
            raise ValueError("invalid expert store format or tensor count")
        if (root / "archer_param_0").stat().st_size != manifest["bytes"]:
            raise ValueError("truncated expert store")
        if hashlib.sha256((root / "archer_index").read_bytes()).hexdigest() != manifest["index_sha256"]:
            raise ValueError("expert index checksum mismatch")
        return manifest
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"incomplete expert store at {root}; use a fresh destination")
    def tensors():
        for shard in sorted(set(weight_map.values())):
            with safe_open(str(Path(model_path) / shard), framework="pt", device="cpu") as f:
                for name in sorted(f.keys()):
                    match = EXPERT.fullmatch(name)
                    if match:
                        layer, expert, projection = match.groups()
                        tid = (int(layer) * config.num_experts + int(expert)) * 3 + PROJECTIONS[projection]
                        yield tid, f.get_tensor(name)
            print(f"prepared {shard}", flush=True)
    manifest = write_expert_store(root, tensors(), identity)
    if manifest["tensors"] != config.num_hidden_layers * config.num_experts * 3:
        raise ValueError("checkpoint does not contain the expected Qwen3 experts")
    return manifest


def load_qwen3_slots(model_path, offload_dir, extension_path=None):
    manifest = prepare_checkpoint_store(model_path, offload_dir)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen3_moe":
        raise ValueError("only Qwen3 MoE is supported by this loader")
    config._attn_implementation = "eager"
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    # Experts execute exclusively in C++; keeping meta placeholders avoids
    # allocating a second model-sized set of expert tensors on any GPU.
    _, weight_map = checkpoint_identity(model_path)
    for shard in sorted(set(weight_map.values())):
        names = [name for name, file in weight_map.items() if file == shard and not EXPERT.fullmatch(name)]
        if not names:
            continue
        with safe_open(str(Path(model_path) / shard), framework="pt", device="cpu") as f:
            for name in names:
                set_module_tensor_to_device(model, name, "cuda:0", value=f.get_tensor(name), dtype=torch.bfloat16)
    # Rotary frequencies are nonpersistent buffers, absent from checkpoints.
    for module in model.modules():
        if module.__class__.__name__ == "Qwen3MoeRotaryEmbedding":
            inv_freq, module.attention_scaling = module.rope_init_fn(config, device=torch.device("cuda:0"))
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.original_inv_freq = inv_freq
    for layer, block in enumerate(model.model.layers):
        block.mlp.layer_id = layer
    os.environ.setdefault("MOE_IO_THREADS", "4")
    os.environ.setdefault("MOE_INFINITY_BUFFERED_IO", "1")
    os.environ.setdefault("MOE_INFINITY_DISABLE_MLOCK", "1")
    extension = load_slot_extension(extension_path)
    handle = extension.prefetch_handle(str(offload_dir), 0.9)
    topology = [(f"layer{l}.experts", [[(l * config.num_experts + e) * 3 + i for i in range(3)]
                                     for e in range(config.num_experts)])
                for l in range(config.num_hidden_layers)]
    handle.set_topology(topology)
    dispatcher = extension.expert_dispatcher(config.num_experts, config.num_hidden_layers, 0, 5, 1)
    for layer, (_, experts) in enumerate(topology):
        for expert, ids in enumerate(experts):
            dispatcher.register_expert(layer, expert, ids, "")
    model.eval()
    return model, handle, dispatcher, manifest
