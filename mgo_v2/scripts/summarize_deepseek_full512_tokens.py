"""Compare the exact 512-token DeepSeek target prefix across four systems."""
import json
from pathlib import Path


ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
REPORT = Path(__file__).resolve().parents[1] / 'experiments/main_table_2x2_20261008'
LABELS = {
    'main_OURS': 'mt2_deepseek_sharegpt_b16_l512_ours_r3_v1',
    'DeepSpeed': 'mt2_deepseek_sharegpt_b16_l512_deepspeed_r3_v2',
    'MoE-Infinity': 'mt2_deepseek_sharegpt_b16_l512_infinity_r3_v1',
    'llama.cpp': 'mt2_deepseek_sharegpt_b16_l512_llama_r3_v1',
}
REFERENCE = ROOT / 'mt2_deepseek_sharegpt_b16_l512_stock512_reference_v1/reference.json'


def repeat_tokens(system, job, repeat):
    filename = f'repeat{repeat}_rank0.json' if system in ('main_OURS', 'DeepSpeed') else f'repeat{repeat}.json'
    row = json.loads((ROOT / job / filename).read_text())
    return dict(zip(row['request_ids'], row['tokens']))


def main():
    reference = json.loads(REFERENCE.read_text())
    assert reference['status'] == 'PASS' and reference['input_tokens'] == 512
    assert reference['local_batch'] == 16 and reference['output_tokens'] == 2
    ids = reference['request_ids']
    expected = dict(zip(ids, reference['tokens']))
    systems = {}
    for name, label in LABELS.items():
        status = json.loads((ROOT / label / 'status.json').read_text())
        assert status['status'] == 'PASS', label
        observed = repeat_tokens(name, label, 1)
        assert all(rid in observed for rid in ids)
        systems[name] = dict(
            job=label,
            first_token_agreement=sum(observed[rid][0] == expected[rid][0] for rid in ids),
            two_token_agreement=sum(observed[rid][:2] == expected[rid] for rid in ids),
            requests=len(ids),
        )
    ds = LABELS['DeepSpeed']
    a, b, c = (repeat_tokens('DeepSpeed', ds, repeat) for repeat in (1, 2, 3))
    systems['DeepSpeed']['repeat_1_2_full64_agreement'] = sum(a[rid] == b[rid] for rid in ids)
    systems['DeepSpeed']['repeat_2_3_full64_agreement'] = sum(b[rid] == c[rid] for rid in ids)
    systems['DeepSpeed']['repeat_1_3_full64_agreement'] = sum(a[rid] == c[rid] for rid in ids)
    output = dict(status='PASS', model='DeepSeek-V2-Lite-Chat', dataset='ShareGPT',
                  local_batch=16, input_tokens=512, reference='stock Transformers BF16, same 16 requests and input length',
                  reference_job=REFERENCE.parent.name, systems=systems,
                  raw_token_ids='retained outside Git in guarded job directories')
    (REPORT / 'DEEPSEEK_FULL512_TOKEN_CHECK.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({name: (row['first_token_agreement'], row['two_token_agreement']) for name, row in systems.items()}))


if __name__ == '__main__':
    main()
