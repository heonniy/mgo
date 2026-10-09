"""Opt-in actual decode: all mandatory H2D completes before grouped math."""
import torch
from . import decode_runtime as module
from .pcie_grouped_probe import AllReadyGrouped


class GroupedDecode:
    def __init__(self,rt):
        assert rt.args.pcie_g2g_first_serial and rt.args.prefetch_off
        self.rt=rt;self.enabled=False;self.validate=False;self.checks=[];self.calls=0
        # Every global token can contribute to all eight selected experts.
        # Keep the historical B16 workspace exactly unchanged; larger batches
        # require a larger activation workspace, never more expert-cache slots.
        self.grouped=AllReadyGrouped(rt.cache,rt.kernel,max_rows=rt.args.local_batch*rt.world*8)
        self.old_pack=module.pack_rank_partial_layout;self.old_compute=rt.compute
        def pack(event,*args,**kwargs):
            prepared=self.grouped.prepare(event) if self.enabled and rt.index>=48 else None
            result=self.old_pack(event,*args,**kwargs)
            if prepared is not None:result['_grouped_plan']=prepared
            return result
        module.pack_rank_partial_layout=pack
        def compute(packet,event,layer):
            if not self.enabled or rt.index<48:return self.old_compute(packet,event,layer)
            slots=event['_grouped_plan']['slots']
            assert all(rt.h2d.ready_many(slots))
            assert all(rt.keys[g[3]]==layer*128+g[0] for g in event['groups'])
            result=self.grouped.all_ready(rt,packet,event,layer);self.calls+=1
            if self.validate and rt.index<96:
                reference=self.old_compute(packet,event,layer)
                ref=torch.cat(reference).float() if reference else rt.cache.new_empty((0,2048)).float()
                alt=torch.cat(result).float() if result else ref
                diff=alt-ref;finite=bool(torch.isfinite(alt).all())
                relative=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(ref).clamp_min(1e-20))
                row=dict(event=rt.index,relative_l2=relative,finite=finite,
                         max_abs=float(diff.abs().max()) if diff.numel() else 0.)
                assert finite and relative<=.01,row
                self.checks.append(row)
            return result
        rt.compute=compute
    def reset(self,enabled,validate=False):
        self.enabled=enabled;self.validate=validate;self.checks=[];self.calls=0
    def receipt(self):
        return dict(enabled=self.enabled,decode_calls=self.calls,workspace_bytes=self.grouped.workspace_bytes,
                    all_ready=True,expert_h2d_compute_overlap=False,numerical_relative_l2_limit=.01,checks=self.checks)
    def close(self):
        module.pack_rank_partial_layout=self.old_pack;self.rt.compute=self.old_compute
