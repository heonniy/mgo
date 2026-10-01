"""Load only the slot data plane; never construct legacy Python controllers."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def load_slot_extension(path: str | None = None):
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("pin CUDA visibility to one device before loading the slot extension")
    if os.environ.get("MOE_EP_DISABLE_ARCHER_EVICT") != "1":
        raise RuntimeError("MOE_EP_DISABLE_ARCHER_EVICT=1 is required")
    if path is None:
        root = Path(__file__).resolve().parents[2] / "MoE-Infinity-EP-archer-coslot" / "moe_infinity"
        candidates = list(root.glob("_store*.so"))
        if len(candidates) != 1:
            raise RuntimeError(f"build the slot extension first; found {len(candidates)} binaries in {root}")
        path = str(candidates[0])
    spec = importlib.util.spec_from_file_location("_store", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_experts(extension, offload_dir, weights, num_layers, num_experts):
    """Create rank-local host backing from (layer, expert, [gate, up, down]).

    This helper is for isolated correctness fixtures. The model loader uses
    a checkpoint-bound manifest for reusable, read-only offload artifacts.
    """
    from .model_loader import write_expert_store

    weights = list(weights)
    os.environ.setdefault("MOE_INFINITY_BUFFERED_IO", "1")
    write_expert_store(offload_dir,
        (((layer * num_experts + expert) * 3 + i, tensor)
         for layer, expert, tensors in weights for i, tensor in enumerate(tensors)),
        {"fixture": True, "num_layers": num_layers, "num_experts": num_experts})
    handle = extension.prefetch_handle(str(offload_dir), 0.9)
    topology = [(f"layer{layer}.experts", [[] for _ in range(num_experts)])
                for layer in range(num_layers)]
    for layer, expert, tensors in weights:
        ids = [(layer * num_experts + expert) * 3 + i for i in range(3)]
        topology[layer][1][expert] = ids
    handle.set_topology(topology)
    # DTYPE_BFLOAT16=0, DEEPSEEK_MOE_DENSE_ACT_DENSE=5 (gate/up/down).
    dispatcher = extension.expert_dispatcher(num_experts, num_layers, 0, 5, 1)
    for layer, (_, groups) in enumerate(topology):
        for expert, ids in enumerate(groups):
            dispatcher.register_expert(layer, expert, ids, "")
    return handle, dispatcher
