"""Active Qwen3-MoE geometry for the main_OURS runtime.

One process serves one model. The model is chosen by ``MGO_MODEL_PATH``
(set by scripts/run_headline_job.py from the workload cell's ``model_path``);
without it the Qwen3-30B-A3B checkpoint is used, so every legacy 30B path keeps
its original constants. All shapes are read from the HF config and validated.
"""
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

QWEN3_30B_PATH = '/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
QWEN3_30B_GEOMETRY = (48, 2048, 768, 128, 8)


@dataclass(frozen=True)
class ModelSpec:
    path: str
    layers: int
    hidden: int
    moe_intermediate: int
    experts: int
    topk: int
    kv_heads: int
    head_dim: int

    @property
    def projection_elements(self):
        return self.hidden * self.moe_intermediate

    @property
    def expert_elements(self):
        return 3 * self.projection_elements

    @property
    def expert_bytes(self):
        return 2 * self.expert_elements

    @property
    def keys(self):
        return self.layers * self.experts

    @property
    def geometry(self):
        return (self.layers, self.hidden, self.moe_intermediate, self.experts, self.topk)

    @property
    def is_qwen3_30b(self):
        return self.geometry == QWEN3_30B_GEOMETRY

    def expert_views(self, row):
        """gate (I,H), up (I,H), down (H,I) views of one flat BF16 slot row."""
        p = self.projection_elements
        return (row[:p].view(self.moe_intermediate, self.hidden),
                row[p:2 * p].view(self.moe_intermediate, self.hidden),
                row[2 * p:].view(self.hidden, self.moe_intermediate))

    def kv_bytes(self, batch, positions, element_size=2):
        return 2 * self.layers * self.kv_heads * self.head_dim * element_size * batch * positions

    def receipt(self):
        return dict(path=self.path, layers=self.layers, hidden=self.hidden,
                    moe_intermediate=self.moe_intermediate, experts=self.experts,
                    topk=self.topk, kv_heads=self.kv_heads, head_dim=self.head_dim,
                    expert_bytes=self.expert_bytes, keys=self.keys)


def spec_from_config(path):
    path = str(Path(path))
    c = json.loads((Path(path) / 'config.json').read_text())
    if c.get('model_type') != 'qwen3_moe':
        raise ValueError(f'unsupported model type {c.get("model_type")!r}')
    if c.get('decoder_sparse_step', 1) != 1 or c.get('mlp_only_layers'):
        raise ValueError('every decoder layer must be sparse MoE')
    if c.get('norm_topk_prob') is not True:
        raise ValueError('runtime routing normalizes top-k probabilities')
    spec = ModelSpec(path=path, layers=int(c['num_hidden_layers']), hidden=int(c['hidden_size']),
                     moe_intermediate=int(c['moe_intermediate_size']),
                     experts=int(c['num_experts']), topk=int(c['num_experts_per_tok']),
                     kv_heads=int(c['num_key_value_heads']),
                     head_dim=int(c.get('head_dim') or c['hidden_size'] // c['num_attention_heads']))
    # Placement state, metadata packing and native kernels assume 128 experts,
    # top-8 routing (uint8 ids, int16 owner bitmasks over <=8 ranks).
    if spec.experts != 128 or spec.topk != 8:
        raise ValueError('main_OURS runtime supports 128 experts with top-8 routing')
    return spec


@lru_cache(None)
def active_spec():
    path = os.environ.get('MGO_MODEL_PATH', QWEN3_30B_PATH)
    spec = spec_from_config(path)
    if Path(path).resolve() == Path(QWEN3_30B_PATH).resolve() and not spec.is_qwen3_30b:
        raise ValueError('unexpected Qwen3-30B geometry')
    return spec


SPEC = active_spec()
