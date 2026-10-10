"""GPU check: inline batch demand submission matches the staging-thread path.

Run: python scripts/test_inline_demand_h2d.py  (needs one idle GPU)
"""
import torch
from mgo_v2.pinned_h2d import PriorityH2DScheduler

N = 9 * 2**20 // 2


def make_sources(n):
    return [torch.full((N,), float(i + 1), dtype=torch.bfloat16).pin_memory() for i in range(n)]


def main():
    torch.cuda.set_device(0)
    src = make_sources(6)
    for inline in (False, True):
        cache = torch.zeros((8, N), dtype=torch.bfloat16, device='cuda')
        q = PriorityH2DScheduler(cache, profile=True, direct_pinned=True)
        # Layer 1: four copies; then compute uses slots 0 and 1.
        items = [(s, 100 + s, [src[s]]) for s in range(4)]
        if inline:
            q.enqueue_demand_batch(items, event_index=1)
        else:
            for slot, key, t in items:
                q.enqueue_demand(slot, key, t, event_index=1)
        q.wait_slots([0, 1, 2, 3], host=True)
        assert all(torch.all(cache[s] == s + 1) for s in range(4))
        x = cache[0].float().sum() + cache[1].float().sum()  # compute reading slots 0/1
        q.record_slots_use([0, 1])
        # Layer 2: overwrite slots 0/1 (must wait for that compute) and a dedup hit on slot 2.
        items = [(0, 200, [src[4]]), (1, 201, [src[5]]), (2, 102, [src[2]])]
        if inline:
            q.enqueue_demand_batch(items, event_index=2)
        else:
            for slot, key, t in items:
                q.enqueue_demand(slot, key, t, event_index=2)
        q.wait_slots([0, 1, 2], host=True)
        assert torch.all(cache[0] == 5) and torch.all(cache[1] == 6) and torch.all(cache[2] == 3)
        assert q.tickets[0].previous_compute is not None and q.tickets[0].previous_copy is not None
        assert q.metrics['copies'] == 6 and q.metrics['urgent_copies'] == 6, q.metrics
        assert len(q.trace) == 6 and q.pending == 0
        float(x)
        q.synchronize()
        delays = [t.submitted_at - t.queued_at for t in q.trace]
        q.close()
        print(f'inline={inline} PASS copies={q.metrics["copies"]} '
              f'max enqueue->submit {1e3 * max(delays):.3f} ms')


if __name__ == '__main__':
    main()
