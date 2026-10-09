"""Check the full-size Qwen3 leaf setup without allocating model weights/GPU memory."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import deepspeed
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from deepspeed.utils.z3_leaf_module import set_z3_leaf_modules
from pcie_deepspeed_leaf import mark_qwen3_moe_leaves


def main(a):
    cfg = Qwen3MoeConfig.from_pretrained(str(a.model), local_files_only=True)
    class Skeleton(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config = cfg
            with torch.device('meta'):
                self.layers = torch.nn.ModuleList([Qwen3MoeSparseMoeBlock(cfg).to(dtype=torch.bfloat16) for _ in range(cfg.num_hidden_layers)])
    receipt = mark_qwen3_moe_leaves(Skeleton(), 17392730112 // 4)
    torch.manual_seed(42)
    tiny = Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(hidden_size=8, moe_intermediate_size=4, num_experts=2, num_experts_per_tok=1)).eval()
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        before = tiny(x)
        set_z3_leaf_modules(tiny, ['Qwen3MoeSparseMoeBlock'])
        after = tiny(x)
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert not torch.cuda.is_initialized()
    import inspect
    paths = [a.model / 'config.json', Path(__file__), Path(__file__).with_name('pcie_deepspeed_leaf.py'),
             Path(inspect.getfile(set_z3_leaf_modules)), Path(inspect.getfile(Qwen3MoeSparseMoeBlock))]
    receipt.update(cpu_only=True, cuda_initialized=False, deepspeed_version=deepspeed.__version__,
        scope='Full-size 48-block BF16 meta skeleton plus exact CPU output/router-logit parity after leaf marking; no distributed GPU execution claimed',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, unix=time.time())
    a.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', blocks=receipt['blocks'], leaf_bytes=receipt['leaves'][0]['full_parameter_bytes'], cuda_initialized=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('/data2/esjung/models/Qwen3-30B-A3B-Instruct-2507'))
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
