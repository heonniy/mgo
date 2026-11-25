"""Stock-HF greedy reference for a Qwen3-MoE checkpoint too large for the GPUs.

The unmodified transformers Qwen3Moe model runs on ONE GPU: dense weights are
resident; each layer's routed experts are read directly from the safetensors
shards (independent of the shared expert store) right before that layer's
stock Qwen3MoeSparseMoeBlock runs, and dropped afterwards. Only the experts
the stock router selects are loaded; the HF block itself recomputes routing.
Output: greedy tokens and top-2 logit margins per request, for comparison with
main_OURS repeat tokens (same request ids, same input tokens).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

EXPERT = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')


def main(a):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_path = Path(a.model)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = 'sdpa'
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    weight_map = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
    handles = {s: safe_open(str(model_path / s), framework='pt', device='cpu')
               for s in sorted(set(weight_map.values()))}
    started = time.time()
    for name, shard in weight_map.items():
        if not EXPERT.fullmatch(name):
            set_module_tensor_to_device(model, name, 'cuda:0', value=handles[shard].get_tensor(name),
                                        dtype=torch.bfloat16)
    for module in model.modules():
        if module.__class__.__name__ == 'Qwen3MoeRotaryEmbedding':
            inv, module.attention_scaling = module.rope_init_fn(config, device=torch.device('cuda:0'))
            module.register_buffer('inv_freq', inv, persistent=False)
            module.original_inv_freq = inv
    dense_seconds = time.time() - started
    stats = dict(expert_loads=0, expert_load_seconds=0.0)

    def hooks(layer, block):
        loaded = []

        def pre(module, args):
            hidden = args[0]
            flat = hidden.reshape(-1, hidden.shape[-1])
            probs = torch.softmax(module.gate(flat), dim=-1, dtype=torch.float32)
            experts = torch.unique(torch.topk(probs, module.top_k, dim=-1).indices).tolist()
            t0 = time.time()
            for e in experts:
                for part in ('gate_proj', 'up_proj', 'down_proj'):
                    name = f'model.layers.{layer}.mlp.experts.{e}.{part}.weight'
                    set_module_tensor_to_device(model, name, 'cuda:0',
                                                value=handles[weight_map[name]].get_tensor(name),
                                                dtype=torch.bfloat16)
            loaded[:] = experts
            stats['expert_loads'] += len(experts)
            stats['expert_load_seconds'] += time.time() - t0

        def post(module, args, output):
            for e in loaded:
                for part in ('gate_proj', 'up_proj', 'down_proj'):
                    set_module_tensor_to_device(model, f'model.layers.{layer}.mlp.experts.{e}.{part}.weight', 'meta')
            loaded.clear()

        block.register_forward_pre_hook(pre)
        block.register_forward_hook(post)

    for layer, decoder in enumerate(model.model.layers):
        hooks(layer, decoder.mlp)
    model.eval()
    rows = json.loads(Path(a.requests).read_text())['requests']
    chosen = [rows[i] for i in a.indices]
    ids = torch.tensor([r['input_ids'][-a.last_tokens:] if a.last_tokens else r['input_ids'] for r in chosen],
                       device='cuda')
    mask = torch.ones_like(ids)
    input_tokens = int(ids.shape[1])
    past = None
    tokens, margins = [], []
    gen_started = time.time()
    with torch.inference_mode():
        for step in range(a.new_tokens):
            pos = torch.arange(mask.shape[1] - ids.shape[1], mask.shape[1], device='cuda')[None, :].expand(len(ids), -1)
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=past,
                        use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            logits = out.logits[:, -1].float()
            top = torch.topk(logits, 2, dim=-1)
            tokens.append(top.indices[:, 0])
            margins.append((top.values[:, 0] - top.values[:, 1]).tolist())
            ids = top.indices[:, :1]
            mask = torch.cat((mask, mask.new_ones((len(ids), 1))), 1)
            print(json.dumps(dict(step=step, seconds=time.time() - gen_started)), flush=True)
    out = dict(status='PASS', model=str(model_path), method='stock transformers Qwen3MoeForCausalLM, '
               'single GPU, routed experts streamed per layer from safetensors (router-selected only)',
               transformers=__import__('transformers').__version__, torch=torch.__version__,
               requests=a.requests, indices=a.indices, request_ids=[r['request_id'] for r in chosen],
               input_tokens=input_tokens,
               batch=len(chosen), tokens=torch.stack(tokens, 1).tolist(),
               top2_margin=[list(x) for x in zip(*margins)], dense_load_seconds=dense_seconds,
               generate_seconds=time.time() - gen_started, **stats,
               peak_allocated_bytes=torch.cuda.max_memory_allocated())
    Path(a.output).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('top2_margin',)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--requests', required=True, help='target/warmup manifest with requests[].input_ids')
    p.add_argument('--indices', type=int, nargs='+', required=True)
    p.add_argument('--last-tokens', type=int, default=0, help='0 = full prompt; 32 = smoke prompts')
    p.add_argument('--new-tokens', type=int, default=4)
    p.add_argument('--output', required=True)
    main(p.parse_args())
