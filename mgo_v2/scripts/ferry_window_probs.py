"""Opt-in raw router-probability recorder for the FERRY W-sensitivity appendix.

Enabled only when MGO_FERRY_WINDOW_PROBS=1 and the eviction-trace capture is
active (see main_eviction_trace.RoutingCapture). Default behaviour is unchanged.

Each rank writes its OWN local rows (no extra collectives, no routing change):
  decode[e, b, :]       float32 rows that LiveMetadata would send for decode
                        layer-event e (=probs[-min(B,128):].float(); B<=128 so all
                        local tokens, in local token order)
  prefill_tail[l, t, :] last PREFILL_TAIL local prefill rows of layer l (float32)
  prefill_sum[l, :]     float64 sum over ALL local prefill rows of layer l
  prefill_count[l]      number of local prefill rows of layer l
Rank-concatenation order (rank0..rank3) is reconstructed offline.
"""
import hashlib, json
import numpy as np

PREFILL_TAIL = 1024


class ProbRecorder:
    def __init__(self, rt, events, layers=48, experts=128):
        self.rt = rt; self.layers = layers; self.experts = experts
        self.batch = rt.args.local_batch; self.tail = min(self.batch, 128)
        assert self.tail == self.batch, 'decode capture assumes the whole local batch is sent'
        self.decode = np.zeros((events - layers, self.batch, experts), np.float32)
        self.prefill_tail = np.zeros((layers, PREFILL_TAIL, experts), np.float32)
        self.prefill_sum = np.zeros((layers, experts), np.float64)
        self.prefill_count = np.zeros(layers, np.int64)
        self.n = 0

    def install(self):
        self.original = self.rt.plan_event
        def plan_event(layer, selected, weights, probs):
            event = self.n
            assert self.rt.index == event and layer == event % self.layers
            p = probs.detach().float()
            if event < self.layers:
                assert p.shape[0] >= PREFILL_TAIL and p.shape[1] == self.experts
                self.prefill_tail[event] = p[-PREFILL_TAIL:].cpu().numpy()
                self.prefill_sum[event] = p.double().sum(0).cpu().numpy()
                self.prefill_count[event] = p.shape[0]
            else:
                assert p.shape == (self.batch, self.experts)
                self.decode[event - self.layers] = p[-self.tail:].contiguous().cpu().numpy()
            self.n += 1
            return self.original(layer, selected, weights, probs)
        self.rt.plan_event = plan_event

    def finish(self, path, rank):
        assert self.n == len(self.decode) + self.layers
        out = path / f'window_probs_rank{rank}.npz'
        np.savez(out, decode=self.decode, prefill_tail=self.prefill_tail,
                 prefill_sum=self.prefill_sum, prefill_count=self.prefill_count)
        h = hashlib.sha256()
        with out.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''): h.update(chunk)
        (path / f'window_probs_rank{rank}.json').write_text(json.dumps(dict(
            status='PASS', rank=rank, local_batch=self.batch, events=self.n,
            prefill_tail=PREFILL_TAIL, dtype='float32', file=out.name,
            file_sha256=h.hexdigest(), bytes=out.stat().st_size), indent=2) + '\n')
        self.rt.plan_event = self.original
