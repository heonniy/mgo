# Env1 bootstrap guard correction

The first attempt stopped before CUDA/process-group initialization because
bootstrap explicitly sets NCCL_P2P_DISABLE=0, while the microbench required
that key to be absent. Accept unset or 0 (P2P enabled), still reject 1 for Env1.
Env2 continues to require NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1.
No timing/calibration or CPU replay result existed from this failed attempt.
Raw attempt retained in microbench_failed_env_check; retry uses a fresh directory.
