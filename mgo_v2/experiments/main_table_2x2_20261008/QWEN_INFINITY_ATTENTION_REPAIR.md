# Qwen MoE-Infinity attention OOM repair

`mt2_qwen_sharegpt_b64_l1024_infinity_r3_v1` failed during warmup prefill. The native MoE-Infinity Qwen loader selected eager attention. With global batch 256 and input length 1024, the eager attention softmax attempted one additional 32 GiB GPU allocation while GPU 0 already held about 59 GiB. The guarded job recorded the failure, released the owner GPUs, and did not count this attempt as a table result.

The Qwen wrapper now selects PyTorch SDPA for all 48 attention layers after MoE-Infinity loads the model. This leaves the model, BF16 weights, EAM cache policy, C30 expert budget, workload, and 64-token timing contract intact; SDPA avoids materializing the full attention matrix. DeepSeek's wrapper is unchanged.

Every earlier Qwen MoE-Infinity PASS row measured with eager attention must be rerun with SDPA before it is retained in the final table. The original attempts remain on disk for audit, and the report should select the newer validated attempts. The two-token smoke and the formerly failing B64/L1024 full run passed with SDPA. Rerun the earlier ShareGPT Qwen MoE-Infinity cells (B16/L512, B64/L512, B16/L1024) before resuming the serial Qwen queue. Subsequent LMSYS Qwen cells use SDPA automatically.
