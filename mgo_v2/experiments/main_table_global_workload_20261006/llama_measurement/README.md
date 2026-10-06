# Native token clock receipt

llama.cpp 3109914090564b4c5280f30896369d55b86bbdbf receives a measurement-only patch: record CLOCK_MONOTONIC microseconds at each sampled token before UTF-8 text buffering, and return the vector in final /completion JSON. This does not change routing, scheduling, sampling, kernels or memory placement.
Use a common client release timestamp on the same monotonic host clock. Global TTFT ends when all requests have token1; E2E ends when all have token64; TPOT=(global end-global first)/63. Final raw token IDs and vector lengths must equal64; prompt token count must match the manifest.
This prevents streamed UTF-8 buffering or HTTP response delivery from being mistaken for model token readiness. Static14GPU-expert-layers/34CPU-expert-layers use1792slots=15.75GiB, <=1843slots.
The measurement-only server build passed. Physical validation remains pending.

EOS consistency: stock llama.cpp `ignore_eos=true` inserts negative-infinity EOG logit biases; the other runtimes continue greedy generation without suppressing logits. Use `ignore_eos=false` plus the scoped `MGO_HEADLINE_CONTINUE_EOG=1` server guard to disable only early termination. This preserves identical greedy/continue semantics and does not alter kernel or scheduling behavior. The native length limit still enforces64 sampled tokens.
