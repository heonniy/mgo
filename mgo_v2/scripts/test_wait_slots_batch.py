"""Check batched host wait with out-of-order tickets and slot reuse."""

import torch

from mgo_v2.pinned_h2d import PriorityH2DScheduler


def main():
    torch.cuda.set_device(0)
    n = 9437184 // 2
    cache = torch.empty((8, n), dtype=torch.bfloat16, device='cuda')
    sources = [torch.full((n,), float(i + 1), dtype=torch.bfloat16,
                          pin_memory=True) for i in range(9)]
    scheduler = PriorityH2DScheduler(cache, direct_pinned=True, profile=True)
    try:
        for slot in range(8):
            scheduler.enqueue_demand(slot, slot, [sources[slot]])
        scheduler.wait_slots([6, 2, 7, 0, 5, 1, 4, 3], host=True)
        assert all(scheduler.ready(slot) for slot in range(8))
        for slot in range(8):
            assert bool((cache[slot] == slot + 1).all())
        # The old compute event and previous copy must still protect reuse.
        scheduler.record_slot_use(0)
        scheduler.enqueue_demand(0, 8, [sources[8]])
        scheduler.wait_slots([0], host=True)
        assert bool((cache[0] == 9).all())
        print('PASS: out-of-order batch completion and slot reuse')
    finally:
        scheduler.close()


if __name__ == '__main__':
    main()
