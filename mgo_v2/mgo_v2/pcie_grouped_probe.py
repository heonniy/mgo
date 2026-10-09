"""Diagnostic paired compute on the SAME R-NEAR packet after isolated H2D.

The native output alone advances the model. Alternate grouped output never
changes routing, cache decisions, or subsequent activations. This is a compute
microdiagnostic, not a grouped serving TPOT result or a Ready-First comparison.
"""
import time
import numpy as np
import torch
import torch.distributed as dist
from . import decode_runtime as module
from .grouped_expert import GroupedExpertExecutor, weight_scatter
import triton as tr


class AllReadyGrouped(GroupedExpertExecutor):
    def prepare(self, event):
        groups = event['groups']  # CPU layout: current event only.
        sizes = [len(g[1]) for g in groups]
        total = sum(sizes)
        if total > self.max_rows:
            raise RuntimeError('Current route exceeds bounded grouped workspace')
        if not groups:
            return dict(sizes=[], slots=[], total=0)
        offsets = np.cumsum([0] + sizes[:-1]).tolist()
        meta = np.asarray([[g[3], offset, count, offset]
                           for g, offset, count in zip(groups, offsets, sizes)], np.int64)
        # All tiny control/index H2D is issued in PLAN preparation, BEFORE
        # metadata completion/forward/H2D isolation. No metadata transfer in GEMM.
        packed = np.concatenate([meta.ravel(), np.concatenate([g[1] for g in groups]),
                                 np.concatenate([g[2] for g in groups])]).astype(np.int64)
        storage = torch.from_numpy(packed).to(self.cache.device)
        end = len(groups)*4
        return dict(sizes=sizes, slots=[g[3] for g in groups], total=total,
                    meta=storage[:end].view(-1,4), rows=storage[end:end+total],
                    cols=storage[end+total:], metadata_bytes=int(packed.nbytes))

    def all_ready(self, rt, packet, event, layer):
        plan = event['_grouped_plan']
        if not plan['sizes']:
            return []
        _, received, _, weights = packet
        total = plan['total']
        torch.index_select(received, 0, plan['rows'], out=self.x[:total])
        self.math(plan['meta'], plan['sizes'], total)
        weight_scatter[(tr.cdiv(max(plan['sizes']),16)*16,len(plan['sizes']))](
            self.y, weights, plan['rows'], plan['cols'], plan['meta'], self.out, weights.shape[1])
        rt.h2d.record_slots_use(plan['slots'])
        return list(self.out[:total].split(plan['sizes']))


class PairedGroupedProbe:
    def __init__(self, rt):
        assert rt.args.pcie_g2g_first_serial and rt.args.prefetch_off
        self.rt=rt;self.enabled=False;self.records=[];self.numerics=[]
        # global64 * topk8 is a strict bound, no expert-weight workspace/D2D.
        self.grouped=AllReadyGrouped(rt.cache,rt.kernel,max_rows=512)
        self.old_pack=module.pack_rank_partial_layout;self.old_compute=rt.compute
        def pack(event,*args,**kwargs):
            prepared=self.grouped.prepare(event) if self.selected(rt.index) else None
            result=self.old_pack(event,*args,**kwargs)
            if prepared is not None:result['_grouped_plan']=prepared
            return result
        module.pack_rank_partial_layout=pack
        def compute(packet,event,layer):
            if not self.selected(rt.index):return self.old_compute(packet,event,layer)
            plan=event['_grouped_plan'];slots=plan['slots']
            assert all(rt.h2d.ready_many(slots))
            assert all(rt.keys[g[3]]==layer*128+g[0] for g in event['groups'])
            # Each packet warms the two methods once outside measured pairs.
            reference=self.old_compute(packet,event,layer)
            alternate=self.grouped.all_ready(rt,packet,event,layer)
            torch.cuda.synchronize()
            ref=torch.cat(reference).float() if reference else rt.cache.new_empty((0,2048)).float()
            alt=torch.cat(alternate).float() if alternate else ref
            diff=alt-ref
            finite=bool(torch.isfinite(alt).all())
            relative=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(ref).clamp_min(1e-20))
            row=dict(event=rt.index,layer=layer,rank=rt.rank,groups=len(slots),rows=plan['total'],
                     max_abs=float(diff.abs().max()) if diff.numel() else 0.,relative_l2=relative,finite=finite)
            self.numerics.append(row)
            assert finite and relative<=.01, row
            for repeat in range(3):
                order=('native','grouped') if (rt.index//3+repeat)%2==0 else ('grouped','native')
                for backend in order:
                    torch.cuda.synchronize();dist.barrier(group=rt.args.pcie_cpu_group)
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start=time.perf_counter_ns();begin.record()
                    values=(self.old_compute(packet,event,layer) if backend=='native'
                            else self.grouped.all_ready(rt,packet,event,layer))
                    end.record();end.synchronize();stop=time.perf_counter_ns()
                    self.records.append(dict(event=rt.index,rank=rt.rank,repeat=repeat,backend=backend,
                                             order=list(order),groups=len(slots),rows=plan['total'],
                                             host_complete_ms=(stop-start)/1e6,cuda_window_ms=begin.elapsed_time(end),
                                             metadata_bytes=plan.get('metadata_bytes',0)))
            # Feed ONLY native math to the model: full original trace is retained.
            return reference
        rt.compute=compute

    def selected(self,event):
        return self.enabled and event//48 in (1,32,63) and event%3==0

    def receipt(self):
        return dict(status='PASS',scope='Paired R-NEAR same-packet compute diagnostic; native output advances model',
                    event_selection='decode steps1,32,63; layers0,3,...45',timed_repeats_per_method_per_event=3,
                    all_experts_ready=True,h2d_compute_overlap=False,workspace_bytes=self.grouped.workspace_bytes,
                    numerical_relative_l2_limit=.01,numerics=self.numerics,records=self.records)

    def close(self):
        module.pack_rank_partial_layout=self.old_pack;self.rt.compute=self.old_compute
