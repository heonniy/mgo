# DeepSeek MoE-Infinity attention repair

The DeepSeek ShareGPT B64/L1024 warmup in `mt2_deepseek_sharegpt_b64_l1024_infinity_r3_v1` failed before any target timing. MoE-Infinity had selected eager attention, and the prefill softmax attempted an additional 16 GiB allocation on GPU 0. The CUDA OOM was outside the expert cache: its immediate cause was the dense attention matrix at the full frozen batch and input length.

The repaired worker selects the model's registered PyTorch SDPA attention interface on all 27 DeepSeek attention layers after MoE-Infinity loads the model. Expert weights, C30 budgets, EAM eviction priorities, disabled speculative admission, request manifests, and three clean target repeats are unchanged. Qwen already used SDPA and retains that path.

To keep the baseline implementation consistent across DeepSeek cells, the report and queue accept a DeepSeek MoE-Infinity PASS only when `attention_backend.json` confirms SDPA on 27 layers. The three earlier ShareGPT DeepSeek Infinity runs with eager attention remain on disk as historical attempts and are remeasured. The failed B64/L1024 attempt remains on disk. The final table selects only new SDPA PASS attempts; no timing sample from an earlier attempt is mixed into them.
