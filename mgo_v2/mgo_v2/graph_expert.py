"""Opt-in B3 executor: exact kernels, stable slot pointers, no timed capture."""
import time
import torch


class GraphExpertExecutor:
    def __init__(self, cache, kernel):
        self.cache = cache
        self.kernel = kernel
        self.signatures = set()
        self.entries = {}
        self.mode = 'discover'
        self.calls = 0
        self.checked = 0
        self.slot_pointer = cache.data_ptr()

    def weights(self, slot):
        w = self.cache[slot]
        return (w[:1572864].view(768, 2048),
                w[1572864:3145728].view(768, 2048),
                w[3145728:].view(2048, 768))

    def __call__(self, slot, x, check=False):
        key = (int(slot), len(x))
        if self.mode == 'discover':
            self.signatures.add(key)
            return self.kernel(x, *self.weights(slot))
        if key not in self.entries:
            raise RuntimeError(f'B3 undiscovered graph signature {key}; run invalid')
        assert self.cache.data_ptr() == self.slot_pointer
        inp, out, graph = self.entries[key]
        inp.copy_(x)
        graph.replay()
        self.calls += 1
        if check:
            expected = self.kernel(x, *self.weights(slot))
            if not torch.equal(out, expected):
                diff = (out.float() - expected.float()).abs()
                raise AssertionError(f'B3 expert parity: max_abs={diff.max().item()}')
            self.checked += 1
        return out

    @torch.inference_mode()
    def build(self, progress=None):
        assert self.mode == 'discover' and self.signatures
        # Inputs and outputs are independent persistent allocations. Only dead
        # within-call temporaries share a graph pool; every graph fully writes
        # its own output before return, and replays are on one current stream.
        required = sum(n * 2048 * 2 * 2 for _, n in self.signatures)
        free, total = torch.cuda.mem_get_info()
        if required > free - 12 * 1024**3:
            raise RuntimeError(f'B3 scratch budget unsafe: required={required}, free={free}')
        torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        started = time.monotonic()
        for i, (slot, n) in enumerate(sorted(self.signatures)):
            if i % 128 == 0:
                free, total = torch.cuda.mem_get_info()
                if free < 10 * 1024**3:
                    raise RuntimeError('B3 graph construction reached reserved VRAM headroom')
                if progress:
                    progress(dict(built=i, total=len(self.signatures),
                                  scratch_bytes=required, elapsed_s=time.monotonic()-started,
                                  free_bytes=free))
            inp = torch.zeros((n, 2048), device=self.cache.device, dtype=torch.bfloat16)
            out = torch.empty_like(inp)
            weights = self.weights(slot)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Warm dynamic-shape variants on the actual slot and shape.
                self.kernel(inp, *weights)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                out.copy_(self.kernel(inp, *weights))
            torch.cuda.current_stream().wait_stream(stream)
            self.entries[(slot, n)] = (inp, out, graph)
        torch.cuda.synchronize()
        self.mode = 'replay'
        self.build_seconds = time.monotonic()-started
        self.scratch_bytes = required

    def receipt(self):
        return dict(mode=self.mode, signatures=sorted(self.signatures),
                    entries=len(self.entries), calls=self.calls, exact_expert_checks=self.checked,
                    scratch_bytes=getattr(self, 'scratch_bytes', None),
                    build_seconds=getattr(self, 'build_seconds', None),
                    weights='live cache slot; no graph-private weight copy')
