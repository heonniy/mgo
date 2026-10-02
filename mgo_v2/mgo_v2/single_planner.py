"""Opt-in rank-zero planner with a fixed-size NCCL decision broadcast."""
from contextlib import nullcontext
from .controller_plan_codec import encode_plan, decode_apply, payload_size


class SinglePlanner:
    def __init__(self, controller, torch, dist, diagnostics=None):
        self.controller, self.torch, self.dist = controller, torch, dist
        self.local_plan = controller.plan_layer
        self.rank = dist.get_rank()
        self.diagnostics = diagnostics
        size = payload_size(controller.config)
        self.host = torch.empty(size, dtype=torch.int32, pin_memory=True)
        self.device = torch.empty(size, dtype=torch.int32, device='cuda')
        self.events = 0
        self.payload_bytes_per_event = size * 4
        controller.plan_layer = self.plan_layer

    def span(self, label):
        return self.diagnostics.span(label) if self.diagnostics is not None else nullcontext()

    def plan_layer(self, routes):
        error = None
        plan = None
        if self.rank == 0:
            try:
                with self.span('planner_compute'):
                    plan = self.local_plan(routes)
                with self.span('plan_encode'):
                    data = encode_plan(plan, self.controller.config, self.controller.cache.tick)
                    self.host.copy_(self.torch.from_numpy(data))
            except Exception as exc:
                error = exc
                self.host.fill_(-1)
        with self.span('planner_broadcast'):
            if self.rank == 0:
                self.device.copy_(self.host)
            self.dist.broadcast(self.device, src=0)
            if self.rank != 0:
                self.host.copy_(self.device)
            self.torch.cuda.current_stream().synchronize()
        if error is not None:
            raise RuntimeError('rank-zero planner failed; failure header broadcast') from error
        if self.rank != 0:
            with self.span('plan_apply'):
                plan = decode_apply(self.host.numpy(), self.controller, routes)
        self.events += 1
        return plan
