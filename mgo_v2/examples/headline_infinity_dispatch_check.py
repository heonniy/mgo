"""Small real BF16 multi-GPU dispatch regression, never a timing sample."""
import argparse
import json
import time
from pathlib import Path

import torch
from moe_infinity import _store


@torch.no_grad()
def main(a):
    assert torch.cuda.device_count() == 4
    torch.set_num_threads(2)
    torch.manual_seed(17)
    store = a.output / 'tiny_store'
    store.mkdir()
    handle = _store.prefetch_handle(str(store), 0.01)
    weights = [[torch.randn(128, 128, dtype=torch.bfloat16) / 16
                for _ in range(3)] for _ in range(4)]
    for e, group in enumerate(weights):
        for i, weight in enumerate(group):
            handle.offload(weight.clone(), e * 3 + i)
    for tid in range(12, 24):
        handle.offload(torch.zeros(1, dtype=torch.bfloat16), tid)
    topology = [(f'dense.before.{i}', [[12 + i]]) for i in range(6)]
    topology += [('model.layers.0.mlp.experts', [list(range(e*3, e*3+3)) for e in range(4)])]
    topology += [(f'dense.after.{i}', [[18 + i]]) for i in range(6)]
    handle.set_topology(topology)
    dispatcher = _store.expert_dispatcher(4, 1, 0, 5, 4)
    for e in range(4):
        dispatcher.register_expert(0, e, list(range(e*3, e*3+3)), '')
    handle.configure_residency_manager(True, True)
    dispatcher.configure_residency_manager(True, True)
    handle.configure_phase_policy(True, 0, 0, 1.0, 1.0, 8)
    for gpu in range(4):
        assert handle.set_expert_budget(gpu, 4 * 3 * 128 * 128 * 2)
    receipts = []
    for rows in (1, 17):
        cpu = torch.randn(rows, 128, dtype=torch.bfloat16) / 4
        for gpu in (1, 3, 0, 2):
            # Leave the caller's default device on zero while tensors and their
            # producer stream belong to another GPU, as real decoder hooks do.
            stream = torch.cuda.Stream(device=gpu)
            with torch.cuda.stream(stream):
                hidden = cpu.to(f'cuda:{gpu}')
                mask = torch.ones(rows, 4, dtype=torch.bool, device=hidden.device)
                scores = torch.full((rows, 4), .25, device=hidden.device)
            torch.cuda.set_stream(stream)
            torch.cuda.set_device(0)
            torch.zeros_like(hidden)
            dispatcher.set_inputs(hidden, mask, scores)
            dispatcher.set_expected_queue(4)
            for e in range(4):
                dispatcher.enqueue_expert(0, e, e, False)
            dispatcher.notify_fetch_start()
            actual = dispatcher.wait_expert()
            for device in range(4):
                torch.cuda.synchronize(device)
            assert actual.device == hidden.device
            expected = torch.zeros_like(actual)
            for group in weights:
                gate, up, down = [w.to(hidden.device) for w in group]
                value = (torch.nn.functional.silu(hidden @ gate.T) * (hidden @ up.T)) @ down.T
                expected += value.float() / 4
            torch.cuda.synchronize(gpu)
            # Fused BF16 activation/product can differ from PyTorch rounding.
            error = (actual - expected).abs().max().item()
            torch.testing.assert_close(actual, expected, atol=.005, rtol=.04)
            receipts.append(dict(rows=rows, input_gpu=gpu, max_abs_error=error))
            print(receipts[-1], flush=True)
    deadline = time.monotonic() + 10
    while dispatcher.get_precision_metrics()['active_leases'] and time.monotonic() < deadline:
        time.sleep(.01)
    assert dispatcher.get_precision_metrics()['active_leases'] == 0
    (a.output / 'result.json').write_text(json.dumps(dict(status='PASS', checks=receipts), indent=2))
    del dispatcher
    handle.clean_up_resources()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cell')
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args())
