# Checkpoint A execution

Use commit 894ed896, R4 GPUs 0/1/4/5, Env1 NCCL_CUMEM_ENABLE=0.
Stop all owned idle-model workers before timing; restore only after workers exit.
No foreign processes are terminated. Model-free buffers plus eight real resident
experts, 25% allocator cap, existing host/GPU free-memory and temperature gates.

A1 freezes three BR-derived distinct packet levels and tests all four synthetic
shapes plus volume-normalized BR/FCA and unchanged raw BR/FCA matrices.
Only remote edges are included; self-copy overhead is outside this isolation.
A2 uses captured GPU sleep plus the actual NCCL all-to-all, with a preceding
GPU all-reduce rendezvous outside measured intervals. The maximum rank-local
completion interval is a synchronized-anchor approximation, not a cross-GPU
absolute timestamp. All achieved delay and per-rank residency samples remain.
No host sleep appears inside measurements. CUDA graphs remove host enqueue
jitter and therefore characterize GPU service rather than whole runtime cost.

A3 compiles the exact expert_kernel AST from env_offload_worker.py, dynamic=True,
fullgraph=True, with actual Qwen3 BF16 weights and runtime tensor shapes.
The power ladder ends at512 because observed decode maximum is510. Additional
row counts calibrate sampled real 2/4/8-expert bundles without interpolation.
Two blocks of100 valid samples, at most a third if median drift exceeds2%;
retain unstable status rather than violating the owner's bounded-repeat rule.
Warmups and compilation are outside samples. No numeric precision sweep.

Publish MICROBENCH_RESULTS and stop at Checkpoint A as explicitly required in
AGENT_TASK.md. Stage B/oracle/online CPA is not launched by this runner.
