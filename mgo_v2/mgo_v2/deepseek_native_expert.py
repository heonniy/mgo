"""DeepSeek-V2-Lite native expert loop over current ready cache slots.

Routing, MAIN/PREFETCH roles, H2D priority and dispatch/return remain in the
existing runtime. No new worker stream, global barrier, weight D2D copy or
grouped GEMM is introduced. Extension compilation must precede measurement.
"""
from functools import lru_cache
from pathlib import Path
import os


@lru_cache(None)
def load_native_expert():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('MAX_JOBS', '1')
    root = Path(__file__).parent / 'csrc'
    return load(name='mgo_deepseek_native_expert_v1',
                sources=[str(root/'native_expert_deepseek.cpp'), str(root/'native_expert.cu')],
                extra_cflags=['-O2'], extra_cuda_cflags=['-O2'], verbose=False)


class DeepseekNativeExpertExecutor:
    def __init__(self, max_experts=64):
        if not 1 <= max_experts <= 64:
            raise ValueError('max_experts must be in [1,64]')
        self.native = load_native_expert()
        self.max_experts = max_experts
        self.waves = self.groups = self.waits = 0

    def compute(self, rt, packet, event, layer):
        mode, received, _, weights = packet
        assert mode == 'current'
        groups = event['groups']
        pending = list(range(len(groups)))
        parts = [None] * len(groups)
        before_wait = 0
        waited = False
        while pending:
            ready = rt.h2d.ready_many([groups[i][3] for i in pending])
            selected = [i for i, ok in zip(pending, ready) if ok][:self.max_experts]
            if not selected:
                # Same dependency as H0: wait for submission on host, then
                # insert a CUDA stream wait. Do not synchronize the host on DMA.
                selected = pending[:1]
                rt.h2d.wait_for_slot(groups[selected[0]][3])
                self.waits += 1
                rt.ready_metrics['waits'] += 1
                if not waited:
                    rt.ready_metrics['ready_before_first_wait'] += before_wait
                waited = True
            elif not waited:
                before_wait += len(selected)
            slots = [int(groups[i][3]) for i in selected]
            assert all(rt.keys[slot] == layer*64+groups[i][0]
                       for i, slot in zip(selected, slots))
            outputs = self.native.execute_wave(rt.cache, received, weights, slots,
                        [groups[i][1] for i in selected], [groups[i][2] for i in selected])
            # Shared completion event conservatively protects every slot in
            # the wave. It is consumed by the unchanged H2D overwrite hazards.
            rt.h2d.record_slots_use(slots)
            for i, output in zip(selected, outputs):
                parts[i] = output
            selected_set = set(selected)
            pending = [i for i in pending if i not in selected_set]
            self.waves += 1
            self.groups += len(selected)
        if not waited:
            rt.ready_metrics['ready_before_first_wait'] += before_wait
        return parts
