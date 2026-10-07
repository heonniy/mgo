"""Untimed stock-HF token check for the new DeepSeek main-table runtime."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL = '/home/hwlee/model/DeepSeek-V2-Lite-Chat'
ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
WORKLOAD = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours', required=True)
    parser.add_argument('--llama', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    manifest = json.loads(WORKLOAD.read_text())
    cell = next(row for row in manifest['cells'] if row['cell'] ==
                'DeepSeekV2Lite_ShareGPT_R4_C30_B16_L512_O64')
    rows = json.loads(Path(cell['target']['path']).read_text())['requests'][:4]
    request_ids = [row['request_id'] for row in rows]
    ids = torch.tensor([row['input_ids'][-32:] for row in rows], device='cuda')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True, device_map='cuda:0').eval()
    tokens = []
    mask = torch.ones_like(ids)
    past = None
    with torch.inference_mode():
        for _ in range(2):
            position = torch.arange(mask.shape[1] - ids.shape[1], mask.shape[1],
                                    device='cuda')[None, :].expand(len(ids), -1)
            output = model(input_ids=ids, attention_mask=mask, position_ids=position,
                           past_key_values=past, use_cache=True, logits_to_keep=1)
            assert torch.isfinite(output.logits).all()
            ids = output.logits[:, -1].argmax(-1)[:, None]
            tokens.append(ids[:, 0].cpu().tolist())
            past = output.past_key_values
            mask = torch.cat((mask, mask.new_ones((len(ids), 1))), 1)
    reference = list(map(list, zip(*tokens)))
    ours = [json.loads((ROOT / args.ours / f'repeat1_rank{rank}.json').read_text())
            for rank in range(4)]
    llama = json.loads((ROOT / args.llama / 'repeat1.json').read_text())
    assert [row['request_ids'][0] for row in ours] == request_ids
    assert llama['request_ids'] == request_ids
    actual_ours = [row['tokens'][0] for row in ours]
    actual_llama = llama['tokens']
    receipt = dict(status='PASS', request_ids=request_ids, stock_hf=reference,
                   main_ours=actual_ours, llama_balanced=actual_llama,
                   ours_first_token_agreement=sum(a[0] == b[0] for a, b in zip(actual_ours, reference)),
                   llama_first_token_agreement=sum(a[0] == b[0] for a, b in zip(actual_llama, reference)))
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in
                      ('status', 'ours_first_token_agreement',
                       'llama_first_token_agreement')}), flush=True)


if __name__ == '__main__':
    main()
