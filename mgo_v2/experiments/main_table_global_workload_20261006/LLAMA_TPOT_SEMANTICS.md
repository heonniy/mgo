# llama.cpp TPOT interpretation correction

Native server uses asynchronous requests/continuous batching. Its existing metric `(max(last)-max(first))/63` is a batch-tail interval, not mean per-request TPOT. Earlier requests can decode before the last request produces its first token. TTFT still means last-request first-token latency and E2E still means all-request completion latency from common client release.

Do not compare the old llama.cpp TPOT column as equivalent to synchronized OURS decode-step latency. Raw timestamps and original values remain preserved. Per-request TPOT includes scheduler/prefill interference, and is not isolated decode kernel time either.

| Job | Repeat | Old batch-tail TPOT s | Mean request TPOT s | First-token spread s |
|---|---:|---:|---:|---:|
| llama_B16_L256_confirmation1 | 1 | 0.257713 | 0.508416 | 36.813 |
| llama_B16_L256_confirmation1 | 2 | 0.257713 | 0.509057 | 37.384 |
| llama_B16_L256_confirmation1 | 3 | 0.261402 | 0.511202 | 37.085 |
| llama_B64_L512_static1 | 1 | 0.404266 | 2.922793 | 342.459 |
| llama_B64_L512_static1 | 2 | 0.403794 | 2.888323 | 333.565 |
| llama_B64_L512_static1 | 3 | 0.415453 | 2.884754 | 329.870 |
