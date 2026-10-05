"""Physical regression: delayed CUDA calls cannot retain scheduler.cv."""
import json
import threading
import torch
from mgo_v2.pinned_h2d import PriorityH2DScheduler


def main():
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    n = 9437184 // 2
    cache = torch.empty((8, n), device='cuda', dtype=torch.bfloat16)
    one = torch.ones(n, dtype=torch.bfloat16)
    two = one * 2
    q = PriorityH2DScheduler(cache, autostart=False)
    first = q.enqueue_prefetch(0, 1, [one])
    entered, release = threading.Event(), threading.Event()
    original = torch.cuda.Event.record
    def delayed(event, stream=None):
        if event is first.done:
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test event release')
        return original(event, stream)
    torch.cuda.Event.record = delayed
    try:
        q.start()
        assert entered.wait(5)
        acquired = q.cv.acquire(timeout=1)
        assert acquired, 'CUDA record retains scheduler lock'
        q.cv.release()
        assert first.state.state == 'INFLIGHT' and not first.submitted
        assert not q.cancel_if_queued(0, 1)
        q.record_slot_use(1)
        q.discard(0, 1)
        second = q.enqueue_demand(0, 2, [two])
        assert second.previous_copy is first.done
    finally:
        release.set()
        torch.cuda.Event.record = original
    q.synchronize()
    assert bool((cache[0] == 2).all()) and q.metrics['copies'] == 2 and q.metrics['canceled'] == 0
    q.wait_for_slot(0)
    before = cache[0].clone()
    q.record_slot_use(0)
    q.enqueue_demand(0, 3, [one])
    q.wait_for_slot(0)
    after = cache[0].clone()
    torch.cuda.synchronize()
    assert bool((before == 2).all()) and bool((after == 1).all())
    q.synchronize()
    ticket = q.tickets[0]
    entered.clear(); release.clear()
    original_query = torch.cuda.Event.query
    result = []
    def delayed_query(event):
        if event is ticket.done:
            entered.set()
            if not release.wait(5):raise TimeoutError('test query release')
        return original_query(event)
    torch.cuda.Event.query = delayed_query
    reader = threading.Thread(target=lambda: result.append(q.ready(0)))
    try:
        reader.start()
        assert entered.wait(5)
        acquired = q.cv.acquire(timeout=1)
        assert acquired, 'CUDA query retains scheduler lock'
        q.cv.release()
        q.invalidate(0, 3)
    finally:
        release.set(); reader.join(timeout=5)
        torch.cuda.Event.query = original_query
    assert result == [False], 'stale readiness must not survive invalidation'
    q.close()
    # Queued urgent work still overtakes background, canceled work is skipped,
    # and double staging buffers survive repeated reuse with distinct content.
    q = PriorityH2DScheduler(cache, profile=True, autostart=False)
    for i in range(6):q.enqueue_prefetch(i, i, [one if i%2 else two])
    q.cancel_if_queued(5, 5)
    q.enqueue_demand(6, 6, [two])
    q.start(); q.synchronize()
    assert q.trace[0].key == 6 and q.metrics['copies'] == 6 and q.metrics['canceled'] == 1
    for i in range(5):assert bool((cache[i] == (1 if i%2 else 2)).all())
    q.close()
    print(json.dumps(dict(status='PASS',delayed_record_lock_free=True,delayed_query_lock_free=True,
                         inflight_replacement_hazard_preserved=True,prior_compute_preserved=True,
                         invalidation_during_query=True,urgent_and_cancellation=True,stage_reuse_checks=5)))


if __name__ == '__main__':main()
