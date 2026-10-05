"""Bounded BF16 GPU parity for ordered unique-row accumulation and fallback."""
import json
import torch
from mgo_v2.unique_combine import add_unique_rows_, validate_unique_layout
from mgo_v2.coalesced_return import combine_rank_partials


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    checks = 0
    for n, width in ((1, 13), (128, 2048), (256, 2048)):
        reference = torch.zeros((n, width), dtype=torch.bfloat16, device='cuda')
        actual = reference.clone()
        for count in (0, 1, n, max(1, n // 3), n):
            indices = torch.randperm(n, device='cuda')[:count]
            values = (torch.randn((count, width), device='cuda') * 16).bfloat16()
            reference.index_add_(0, indices, values)
            add_unique_rows_(actual, indices, values)
            assert torch.equal(reference.view(torch.int16), actual.view(torch.int16))
            checks += 1
    class Exchange:
        rank = 0
        return_bytes = 0
        calls = 0
        def exchange(self, partial, send, recv):
            assert sum(send) == sum(recv) == partial.shape[0]
            self.calls += 1
            return partial.clone(), None
    hidden = torch.zeros((5, 2048), dtype=torch.bfloat16, device='cuda')
    for duplicates in (False, True):
        rows = [[0, 1, 2], [1, 3, 4], [0, 4]]
        if duplicates:
            rows[0] = [0, 0, 2]
        cpu = dict(recv_counts=[2, 0, 3, 0], send_counts=[2, 0, 3, 0], send_idx=[0, 1, 0, 2, 4],
                   groups=[(i, r, [0]*len(r), i) for i, r in enumerate(rows)])
        valid = validate_unique_layout(cpu, 5)
        assert valid == (not duplicates)
        event = dict(cpu, unique_combine_validated=valid)
        event['send_idx'] = torch.tensor(cpu['send_idx'], device='cuda')
        event['groups'] = [(i, torch.tensor(r, device='cuda'), c, s) for i, r, c, s in cpu['groups']]
        parts = [torch.randn((len(r), 2048), device='cuda').bfloat16() for r in rows]
        baseline, candidate = Exchange(), Exchange()
        expected = combine_rank_partials(baseline, hidden, parts, event, torch.bfloat16)
        result = combine_rank_partials(candidate, hidden, parts, event, torch.bfloat16, unique_rows=True)
        assert torch.equal(expected.view(torch.int16), result.view(torch.int16))
        assert baseline.calls == candidate.calls == 1 and baseline.return_bytes == candidate.return_bytes
        checks += 1
    assert not validate_unique_layout(dict(cpu, send_idx=[0, 1, 0, 2, 99]), 5)
    torch.cuda.synchronize()
    print(json.dumps(dict(status='PASS', bitwise_checks=checks, precision='bf16',
                         scope='GPU row kernel and simulated return exchange; full model validation pending',
                         peak_bytes=torch.cuda.max_memory_allocated())))


if __name__ == '__main__':
    main()
