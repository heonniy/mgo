"""Round-trip Qwen/DeepSeek fused payload layouts through one-rank Gloo."""
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from mgo_v2.fused_transport import FusedTokenRankTransport


def check(topk):
    hidden = torch.arange(2 * 2048, dtype=torch.float32).reshape(2, 2048).to(torch.bfloat16)
    ids = torch.stack((torch.arange(topk), torch.arange(topk) + 1)).long()
    dense = torch.zeros((2, 128), dtype=torch.bfloat16)
    dense.scatter_(1, ids, .25)
    event = dict(send_idx=torch.tensor([0, 1]), send_eids=ids,
                 send_counts=[2], recv_counts=[2])
    transport = FusedTokenRankTransport('rank-partial', topk=topk)
    packet = transport.forward(hidden, dense, event, async_op=False)
    received, weights, selected = packet.finish()
    assert torch.equal(received, hidden)
    assert weights.shape == selected.shape == (2, topk)
    assert torch.equal(weights, torch.full_like(weights, .25))
    assert torch.equal(selected.long(), ids)
    assert transport.calls == 1


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group('gloo', init_method='file://' + str(Path(directory) / 'init'),
                                rank=0, world_size=1)
        try:
            check(8)
            check(6)
        finally:
            dist.destroy_process_group()
    print('PASS Qwen and DeepSeek fused payloads')
