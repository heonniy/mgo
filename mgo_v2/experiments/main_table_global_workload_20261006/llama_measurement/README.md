# Native token clock receipt

llama.cpp 3109914090564b4c5280f30896369d55b86bbdbf receives a measurement-only patch: record CLOCK_MONOTONIC microseconds at each sampled token before UTF-8 text buffering, and return the vector in final /completion JSON. This does not change routing, scheduling, sampling, kernels or memory placement.
Use a common client release timestamp on the same monotonic host clock. Global TTFT ends when all requests have token1; E2E ends when all have token64; TPOT=(global end-global first)/63. Final raw token IDs and vector lengths must equal64; prompt token count must match the manifest.
This prevents streamed UTF-8 buffering or HTTP response delivery from being mistaken for model token readiness. Static14GPU-expert-layers/34CPU-expert-layers use1792slots=15.75GiB, <=1843slots.
The measurement-only server build passed. Physical validation remains pending.
