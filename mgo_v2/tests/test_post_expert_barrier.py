"""Static guard for optional post-expert global barrier isolation mode.

CPU-only: verifies ordering and case plumbing without importing CUDA workers.
"""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=(ROOT/'mgo_v2/decode_runtime.py').read_text()
SELECTED=(ROOT/'mgo_v2/selected_runtime.py').read_text()
B3=(ROOT/'examples/b3_measure_worker.py').read_text()


def between(text,left,right):
    lo=text.index(left);hi=text.index(right,lo)
    return text[lo:hi]


def test_post_expert_barrier_order_and_plumbing():
    execute=between(RUNTIME," def execute(self,*args):","  if self.index==48:")
    expert=execute.index("with nvtx_phase('moe.expert_compute')")
    local=execute.index("with nvtx_phase('moe.post_expert_local_complete')")
    barrier=execute.index("with nvtx_phase('moe.post_expert_global_barrier')")
    ret=execute.index("with nvtx_phase('moe.return_a2a')")
    assert expert < local < barrier < ret
    block=between(execute,"with nvtx_phase('moe.post_expert_local_complete')","   if fused:")
    assert "torch.cuda.current_stream().synchronize()" in block
    assert "dist.barrier()" in block
    assert "self.post_expert_barriers+=1" in block
    assert "getattr(self.args,'post_expert_barrier',False)" in execute

    assert "options['post_expert_barrier']=case.get('post_expert_barrier',False)" in SELECTED
    assert "Invalid post-expert barrier flag" in SELECTED
    assert "'post_expert_barrier'" in B3
    assert "expected_barriers=horizon*48 if first.get('post_expert_barrier',False) else 0" in B3
