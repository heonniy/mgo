"""Untimed stock BF16 reference for the full-length DeepSeek table cell."""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL = '/home/hwlee/model/DeepSeek-V2-Lite-Chat'


def main(args):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,4,5'
    manifest = json.loads(Path(os.environ['MGO_HEADLINE_WORKLOADS']).read_text())
    cell = next(row for row in manifest['cells'] if row['cell'] == args.cell)
    assert cell['model'] == 'DeepSeekV2Lite' and cell['input_tokens'] == 512
    assert cell['local_batch'] == 16
    rows = json.loads(Path(cell['target']['path']).read_text())['requests'][:16]
    assert len(rows) == 16 and all(len(row['input_ids']) == 512 for row in rows)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True, device_map='cuda:0').eval()
    ids = torch.tensor([row['input_ids'] for row in rows], device='cuda:0')
    mask = torch.ones_like(ids)
    past = None
    generated = []
    with torch.inference_mode():
        for _ in range(2):
            out = model(input_ids=ids, attention_mask=mask,
                        past_key_values=past, use_cache=True, logits_to_keep=1)
            assert torch.isfinite(out.logits).all()
            ids = out.logits[:, -1].argmax(-1)[:, None]
            generated.append(ids[:, 0].cpu().tolist())
            past = out.past_key_values
            mask = torch.cat((mask, mask.new_ones((len(ids), 1))), dim=1)
    tokens = list(map(list, zip(*generated)))
    receipt = dict(status='PASS', cell=args.cell, model_path=MODEL,
                   input_tokens=512, local_batch=16, output_tokens=2,
                   request_ids=[row['request_id'] for row in rows], tokens=tokens,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    (args.output / 'reference.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output / 'result.json').write_text(json.dumps(dict(status='PASS', kind='untimed_stock_reference')) + '\n')
    print(json.dumps(dict(status='PASS', requests=len(rows), input_tokens=512)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
