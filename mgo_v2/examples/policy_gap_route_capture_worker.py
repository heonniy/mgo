"""Capture an input128 prefill-gate routing proxy for seed nomination."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

MODEL = '/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(args):
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,4,5'
    manifest_path = Path(os.environ['MGO_GAP_POOL_MANIFEST'])
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'PASS' and manifest['pool_size'] >= 512
    requests = manifest['requests']
    assert all(len(row['input_ids']) == 128 for row in requests)
    device = torch.device(f'cuda:{local}')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85, device=device)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
                                                  device_map={'': str(device)}, low_cpu_mem_usage=True,
                                                  attn_implementation='sdpa').eval()
    indices = np.arange(rank, len(requests), 4, dtype=np.int32)
    routes = np.empty((48, len(indices), 8), dtype=np.uint8)
    active = dict(batch=0, route=None, seen=0)
    hooks = []
    for layer, module in enumerate(model.model.layers):
        def capture(_module, _inputs, logits, layer=layer):
            batch = active['batch']
            assert logits.shape == (batch * 128, 128)
            selected = logits.reshape(batch, 128, 128)[:, -1].topk(8, dim=-1).indices
            active['route'][layer] = selected.to(torch.uint8).cpu().numpy()
            active['seen'] += 1
        hooks.append(module.mlp.gate.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            for start in range(0, len(indices), 16):
                subset = indices[start:start + 16]
                free, total = torch.cuda.mem_get_info(device)
                if free < 8 * 2**30:
                    raise RuntimeError(f'GPU memory guard: free={free} total={total}')
                ids = torch.tensor([requests[int(i)]['input_ids'] for i in subset], device=device)
                active.update(batch=len(subset), route=np.empty((48, len(subset), 8), dtype=np.uint8), seen=0)
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
                torch.cuda.synchronize(device)
                assert active['seen'] == 48
                routes[:, start:start + len(subset)] = active['route']
                if rank == 0:
                    write(args.output / 'phase.json', dict(phase='proxy_capture', completed=start + len(subset),
                                                           local_total=len(indices), dataset=manifest['dataset']))
        np.save(args.output / f'proxy_rank{rank}.npy', routes)
        np.save(args.output / f'indices_rank{rank}.npy', indices)
        write(args.output / f'receipt_rank{rank}.json', dict(status='PASS', rank=rank, batches=(len(indices) + 15) // 16,
                                                             requests=len(indices), model=MODEL))
    finally:
        for hook in hooks:
            hook.remove()
    dist.barrier()
    if rank == 0:
        proxy = np.empty((48, len(requests), 8), dtype=np.uint8)
        for other in range(4):
            order = np.load(args.output / f'indices_rank{other}.npy')
            proxy[:, order] = np.load(args.output / f'proxy_rank{other}.npy')
        destination = manifest_path.parent / 'LMSYS_proxy_routes_input128.npy'
        np.save(destination, proxy)
        write(args.output / 'result.json', dict(status='PASS', dataset=manifest['dataset'],
                                                manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                                proxy=str(destination), shape=list(proxy.shape),
                                                definition='Last prefill token, top8 gate logits per request/layer; seed-screen proxy only'))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args())
