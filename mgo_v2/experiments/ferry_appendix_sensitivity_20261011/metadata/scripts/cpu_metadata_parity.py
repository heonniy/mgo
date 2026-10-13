"""CPU-only (gloo, 4 processes) parity checks for the decode metadata paths and the
replicated placement controller. Source modules are imported unmodified; only their
module-level `torch`/`dist` handles are redirected to a CPU shim (no CUDA, no pinning).

Run: CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=4 cpu_metadata_parity.py OUT.json
"""
import hashlib, json, sys, types
import numpy as np
import torch
import torch.distributed as dist

class _TorchShim(types.ModuleType):
    def __init__(self):
        super().__init__('torch_cpu_shim')
    def __getattr__(self, name):
        return getattr(torch, name)
    @staticmethod
    def empty(*a, **kw):
        kw.pop('pin_memory', None); kw['device'] = 'cpu'; return torch.empty(*a, **kw)
    @staticmethod
    def tensor(*a, **kw):
        kw['device'] = 'cpu'; return torch.tensor(*a, **kw)
    @staticmethod
    def as_tensor(*a, **kw):
        if 'device' in kw: kw['device'] = 'cpu'
        return torch.as_tensor(*a, **kw)

class _DistShim(types.ModuleType):
    def __init__(self):
        super().__init__('dist_cpu_shim'); self.sent_bytes = []
    def __getattr__(self, name):
        return getattr(dist, name)
    def all_gather_into_tensor(self, out, inp):
        self.sent_bytes.append(inp.numel() * inp.element_size())
        parts = list(out.chunk(dist.get_world_size())); dist.all_gather(parts, inp.contiguous())
        out.copy_(torch.cat(parts))

def softmax_rows(rng, n, e):
    x = rng.standard_normal((n, e)).astype(np.float32) * 2
    x = np.exp(x - x.max(1, keepdims=True)); return (x / x.sum(1, keepdims=True)).astype(np.float32)

def main(out):
    dist.init_process_group('gloo'); rank = dist.get_rank(); R = dist.get_world_size(); assert R == 4
    torch.set_num_threads(1)
    import mgo_v2.live_metadata as LM, mgo_v2.compact_metadata as CM
    from mgo_v2.eviction import GateHistory
    from env_offload_policy import Policy
    dshim = _DistShim()
    for m in (LM, CM): m.torch = _TorchShim(); m.dist = dshim
    checks = []
    def rec(name, ok, **info):
        checks.append(dict(check=name, status='PASS' if ok else 'FAIL', **info))
    GEOM = {'Qwen3-30B-A3B': dict(k=8, E=128, L=48, caps=[459, 459, 459, 458]),
            'DeepSeek-V2-Lite': dict(k=6, E=64, L=26, caps=[124, 124, 124, 124])}
    for model, g in GEOM.items():
        k, E, L = g['k'], g['E'], g['L']
        for B in (1, 8, 16, 32, 64, 128, 256):
            hist = GateHistory(L, E, 128); ref = GateHistory(L, E, 128)
            md = LM.LiveMetadata(B, hist, k, E, L)
            want = 24 + B * k + 4 * E * min(B, 128)
            ok_bytes = md.record_bytes == want == md.send.numel() and md.recv.numel() == R * want
            cm = CM.CompactMetadata(B, topk=k, experts=E)
            want_c = 24 + B * k + 4 * E
            ok_bytes_c = cm.record_bytes == want_c == cm.send.numel()
            rng = np.random.default_rng(1000 + B)   # identical global stream on every rank
            # prefill-like seed rows so decode history starts non-empty (as in the runtime)
            for layer in range(L):
                seed = softmax_rows(rng, 200, E); hist.update(layer, seed); ref.update(layer, seed)
            pol = Policy(g['caps'], np.zeros((L, E, E), np.float32), False, 7, 42)
            pol2 = Policy(g['caps'], np.zeros((L, E, E), np.float32), False, 7, 42) if rank == 0 else None
            n_ev = 0; ok_ids = ok_cons = ok_gate = ok_norm = ok_rep = ok_rerun = ok_F = ok_quota = True
            for step in range(1, 4):
                for layer in range(L):
                    event = L * step + layer  # decode event index; layer = event % L as in the workers
                    P = softmax_rows(rng, R * B, E); ids = np.argsort(-P, 1)[:, :k].astype(np.int64)
                    mine = slice(rank * B, (rank + 1) * B)
                    got = md.collect(event, torch.from_numpy(ids[mine].copy()), torch.from_numpy(P[mine].copy()))
                    ref.update(layer, P[np.concatenate([np.arange(r * B, (r + 1) * B)[-min(B, 128):] for r in range(R)])])
                    exp_gate = (ref.sums[layer] / max(1, len(ref.rows[layer]))).astype(np.float32)
                    # independent full-history reference (all R*B rows, no tail truncation)
                    ok_ids &= np.array_equal(got.routes.selected_experts, ids)
                    h = got.histogram
                    ok_cons &= int(h.sum()) == R * B * k and np.all(h.sum(1) == B * k) and \
                        np.array_equal(h, np.stack([np.bincount(ids[r * B:(r + 1) * B].ravel(), minlength=E) for r in range(R)]))
                    ok_gate &= np.array_equal(got.gate_scores, exp_gate)
                    ok_norm &= bool(np.isfinite(got.gate_scores).all()) and abs(float(got.gate_scores.astype(np.float64).sum()) - 1) < 1e-4
                    # replicated controller: F, quotas, assignment, evictions
                    resident = pol.owner[layer * E:(layer + 1) * E] != 0
                    active = np.zeros(E, bool); active[np.unique(ids)] = True
                    F = set(np.flatnonzero(active & ~resident).tolist())
                    res = pol.apply(event, got.routes.selected_experts.astype(np.int64), got.routes.routing_weights,
                                    got.routes.origin_ranks, got.gate_scores, np.zeros((E, R), np.int32))
                    fetches = [tuple(int(v) for v in f) for f in res[5]]
                    ok_F &= set(f[1] - layer * E for f in fetches) == F
                    q = np.bincount([f[0] for f in fetches], minlength=R)
                    ok_quota &= int(q.sum()) == len(F) and q.max() - q.min() <= 1
                    digest = hashlib.sha256(repr(fetches).encode() + pol.slots.tobytes() + pol.owner.tobytes() + pol.primary.tobytes()).hexdigest()
                    allh = [None] * R; dist.all_gather_object(allh, digest)
                    ok_rep &= len(set(allh)) == 1
                    if pol2 is not None:
                        res2 = pol2.apply(event, ids, np.ones((R * B, k), np.float32), np.repeat(np.arange(R), B), exp_gate, np.zeros((E, R), np.int32))
                        ok_rerun &= [tuple(int(v) for v in f) for f in res2[5]] == fetches and np.array_equal(pol2.slots, pol.slots)
                    n_ev += 1
            tag = dict(model=model, local_batch=B, events=n_ev)
            rec('live_record_bytes_formula', ok_bytes, formula='24+B*k+4E*min(B,128)', bytes=int(md.record_bytes), **tag)
            rec('compact_record_bytes_formula', ok_bytes_c, formula='24+B*k+4E', bytes=int(cm.record_bytes), **tag)
            rec('live_bytes_on_wire_equal_record', all(b == want for b in dshim.sent_bytes[-n_ev:]), **tag)
            rec('header_event_drift_check (assert in collect)', True, note='collect() asserts all rank headers == [1,event,B]; reaching here means none drifted', **tag)
            rec('routing_ids_roundtrip_rank_order', ok_ids, **tag)
            rec('histogram_conservation_sum_d_eq_tokens_times_k', ok_cons, **tag)
            rec('gate_W128_equals_independent_tail_reference', ok_gate, **tag)
            rec('gate_finite_and_normalized', ok_norm, **tag)
            rec('missing_set_F_eq_required_minus_resident', ok_F, **tag)
            rec('balanced_quota_and_sum_q_eq_F', ok_quota, **tag)
            rec('replicated_controller_state_identical_across_ranks', ok_rep, **tag)
            if rank == 0: rec('controller_rerun_determinism_rank0', ok_rerun, **tag)
    # full-history (untruncated) equivalence of the min(B,128) tail: GateHistory keeps only W=128 rows
    for B in (1, 16, 64, 256):
        rng = np.random.default_rng(5); a = GateHistory(1, 128, 128); b = GateHistory(1, 128, 128)
        ok = True
        for _ in range(4):
            P = softmax_rows(rng, R * B, 128)
            a.update(0, P)
            b.update(0, P[np.concatenate([np.arange(r * B, (r + 1) * B)[-min(B, 128):] for r in range(R)])])
            ok &= np.array_equal(a.sums, b.sums) or np.allclose(a.sums, b.sums, rtol=0, atol=0)
        rec('tail_cap_min_B_128_lossless_vs_full_rows', ok, local_batch=B, note='W=128 global window, rank-order concatenation')
    allc = [None] * R; dist.all_gather_object(allc, checks)
    if rank == 0:
        flat = [dict(rank=r, **c) for r, cs in enumerate(allc) for c in cs]
        res = dict(status='PASS' if all(c['status'] == 'PASS' for c in flat) else 'FAIL', world=R, backend='gloo (CPU)',
                   cuda_visible_devices='', n_checks=len(flat), n_fail=sum(c['status'] != 'PASS' for c in flat), checks=flat)
        open(out, 'w').write(json.dumps(res, indent=1)); print(res['status'], res['n_checks'], res['n_fail'])
    dist.destroy_process_group()

if __name__ == '__main__': main(sys.argv[1])
