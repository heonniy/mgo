# Quiet-host paired frozen-route policy check

The four managed model-inference loads on GPUs 2/3/6/7 were stopped before these jobs; those GPUs were idle. Both jobs used R4 GPUs 0/1/4/5, ShareGPT input128, B8 per rank, 32 decode forwards, C30, P2P-disabled same-host transport, prefetch OFF, and the **same route hash and teacher tokens on all four ranks**. Each policy has two unfiltered cache-reset timings. Full TPOT includes attention.

| Executor | BR TPOT (s/token) | Near TPOT (s/token) | Static TPOT (s/token) | Near vs BR |
|---|---:|---:|---:|---:|
| new_OURS grouped | 0.2622 [0.2618, 0.2626] | 0.2637 [0.2633, 0.2641] | 0.2810 [0.2802, 0.2818] | -0.58% |
| main_OURS Ready-First | 0.4362 [0.4360, 0.4363] | 0.4279 [0.4266, 0.4292] | 0.4403 [0.4400, 0.4405] | +1.90% |

The `main_OURS` Ready-First Near range is wholly below its BR range, whereas grouped `new_OURS` Near is wholly above BR. The executor change therefore reverses the observed policy ordering on this one frozen cell. It does not establish the same ordering for other batches, caches, or transports. Per-run timings, peer/H2D bytes, and raw paths are in [QUIET_PAIR.json](QUIET_PAIR.json).
