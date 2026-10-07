"""Derive C30 expert budgets from checkpoint tensor shapes, without loading weights."""
import json
from pathlib import Path

from safetensors import safe_open


REPO = Path(__file__).resolve().parents[1] / 'experiments/main_table_2x2_20261008'
MODELS = {
    'Qwen3': Path('/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'),
    'DeepSeekV2Lite': Path('/home/hwlee/model/DeepSeek-V2-Lite-Chat'),
}


def main():
    result = {'status': 'PASS', 'models': {}}
    for name, path in MODELS.items():
        config = json.loads((path / 'config.json').read_text())
        index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
        if name == 'Qwen3':
            prefix = 'model.layers.0.mlp.experts.0.'
            layers, experts = config['num_hidden_layers'], config['num_experts']
            expected_shapes = {'gate_proj.weight': [768, 2048], 'up_proj.weight': [768, 2048],
                               'down_proj.weight': [2048, 768]}
            slots = 1843
        else:
            prefix = 'model.layers.1.mlp.experts.0.'
            layers = config['num_hidden_layers'] - config['first_k_dense_replace']
            experts = config['n_routed_experts']
            expected_shapes = {'gate_proj.weight': [1408, 2048], 'up_proj.weight': [1408, 2048],
                               'down_proj.weight': [2048, 1408]}
            slots = 499
        shapes = {}
        for suffix, expected in expected_shapes.items():
            key = prefix + suffix
            shard = path / index[key]
            with safe_open(str(shard), framework='pt', device='cpu') as handle:
                shape = list(handle.get_slice(key).get_shape())
            assert shape == expected, (name, key, shape, expected)
            shapes[key] = shape
        expert_bytes = sum(shape[0] * shape[1] * 2 for shape in shapes.values())
        total = layers * experts
        assert slots == int(total * .30), (name, slots, total)
        result['models'][name] = dict(model_path=str(path), routed_layers=layers,
                                       routed_experts_per_layer=experts, routed_expert_count=total,
                                       expert_bytes=expert_bytes, c30_slots=slots,
                                       c30_expert_budget_bytes=slots * expert_bytes,
                                       weight_shapes=shapes)
    output = REPO / 'MODEL_BUDGET_AUDIT.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'PASS {output}')


if __name__ == '__main__':
    main()
