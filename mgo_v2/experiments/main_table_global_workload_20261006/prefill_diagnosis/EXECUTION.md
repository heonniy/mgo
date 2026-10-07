# Prefill TTFT attribution execution

Owner plan e641419. B64/L512 first, then B16/L256; R4/C30 Near/H0/full-pinned prefill-optimized path only, physical GPUs0/1/4/5.

Each cell runs one clean one-token warmup followed by one cold one-token diagnostic. No decode generation or headline timing repeat. Existing primary results stay unchanged. Native greedy first tokens will be checked against the prior optimized full-generation primary for the same request IDs.

Instrumentation records nested exclusive wall, own-thread CPU and current-stream CUDA-event intervals. It adds no phase barriers or phase synchronization. Metadata collectives are separated by payload; controller/layout methods and existing runtime ranges are wrapped. H2D scheduler records copy-stream intervals independently. Those copy service intervals must not be summed into the current-stream TTFT partition.

Current-stream intervals include host submission gaps, allocations and peer waits; expert service is not pure GEMM kernel-active time. Attention/dense residual includes model work outside MoE and token selection. Global completion is max per-rank first-token time; sum of per-layer rank maxima is not used as an additive critical path.

CPU wall times are a parallel explanatory view, not an additional additive GPU cost. Boundary residual reconciles local TTFT to CUDA timeline; limit50ms. All raw per-layer segments, H2D copies, resources and logits/token/cache guards are preserved.
