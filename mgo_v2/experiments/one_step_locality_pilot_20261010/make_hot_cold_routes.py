"""Freeze a heterogeneous one-step route with known cold hot experts.

This is a synthetic router intervention on real Qwen hidden states/weights,
not an accuracy or workload-representative benchmark.
"""

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path('/home/hwlee/mgo-results/single_step_locality_pilot_20261010')
SOURCE = ROOT / 'jobs/r4_n_br_c30_16_diagnostic_v5'
ROUTE_SOURCE = ROOT / 'jobs/r4_n_br_c30_16_diagnostic_v4'
OUTPUT = ROOT / 'hot_cold_routes'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    diagnostics = [json.loads((SOURCE / f'generation_diagnostic_rank{r}.json').read_text())
                   for r in range(4)]
    source_routes = [torch.load(ROUTE_SOURCE / f'decode_routes_rank{r}.pt',
                                map_location='cpu', weights_only=True) for r in range(4)]
    output_routes = [[] for _ in range(4)]
    digests = [hashlib.sha256() for _ in range(4)]
    hot_by_layer = []
    for layer in range(48):
        # These experts were fetched in the real BR decode after the common
        # prefill, so the synthetic hot choices begin as actual cache misses.
        cold_misses = sorted({copy['key'] % 128 for diag in diagnostics
                              for copy in diag['h2d_copies']
                              if copy['event_index'] == 48 + layer})
        assert len(cold_misses) >= 16
        hot = cold_misses[:16]
        hot_by_layer.append(hot)
        rest = [expert for expert in range(128) if expert not in hot]
        for rank in range(4):
            _, weights, probs = source_routes[rank]['routes'][layer]
            selected = torch.empty((16, 8), dtype=torch.int64)
            for token in range(16):
                selected[token, :4] = torch.tensor(hot[rank * 4:rank * 4 + 4])
                for k in range(4):
                    selected[token, 4 + k] = rest[(rank * 64 + token * 4 + k) % len(rest)]
            weights = torch.full_like(weights, 1 / 8)
            probs = torch.zeros_like(probs)
            probs.scatter_(1, selected, 1 / 8)
            assert torch.allclose(probs.sum(1), torch.ones(16))
            output_routes[rank].append((selected, weights, probs))
            digests[rank].update(selected.to(torch.int16).numpy().tobytes())
    for rank in range(4):
        torch.save(dict(cell=source_routes[rank]['cell'], rank=rank,
                        routes=output_routes[rank]), OUTPUT / f'decode_routes_rank{rank}.pt')
    (OUTPUT / 'manifest.json').write_text(json.dumps(dict(
        status='FROZEN', source=str(SOURCE), physical_gpus=[0, 1, 4, 5],
        local_batch=16, layers=48, route_sha256_by_rank=[x.hexdigest() for x in digests],
        hot_experts_by_layer=hot_by_layer,
        construction='16 real-prefill-missing hot experts/layer, four per rank on every token; four rotating cold experts/token; uniform top8 weights',
    ), indent=2) + '\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
