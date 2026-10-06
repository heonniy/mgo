Primaries1/2 pass exact256-request/64-token coverage, raw clocks, KV weakref release, cold expert residency, EAM/priority activity and all GPU byte caps.

Repeat2: TTFT9.650224808s, TPOT3.637762132381s, E2E238.829239148s. Relative first-two differences: {"TTFT": 0.25344232917397136, "TPOT": 1.593671806536911, "E2E": 1.539866751967408}.

Warmup and both primaries have identical per-GPU PyTorch peak allocated bytes; prior retained-KV extra allocation does not recur. Native allocator bytes are separately charged. Repeat3 pending under the committed three-primary protocol; no final stability claim.
