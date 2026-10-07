# Env2 transport recovery

The first C30/B8 P2P-disabled model attempt (`policy_gap_sharegpt_c30_p2p_disabled_b8_case0_20261008`) failed before warmup completed. All ranks set `NCCL_P2P_DISABLE=1`; NCCL selected `NET/IB`, and the first `ALLREDUCE` reported `IBV_WC_RETRY_EXC_ERR`. Its raw `run.log` and `status.json` remain in `/home/hwlee/mgo-results/headline_r4_20261007/`.

The bounded four-rank dispatch/combine smoke then passed with both `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`, while `NCCL_CUMEM_ENABLE=0` and `NCCL_SHM_DISABLE` remained unset. Four rank receipts passed payload validation, and NCCL INFO reported `SHM/direct/direct`. The smoke receipt is `ENV2_TRANSPORT_SMOKE.json`; the raw log is `/home/hwlee/mgo-results/policy_gap_c60_20261008/env2_shm_smoke_v3/`.

Subsequent env2 model runs use these same three effective NCCL variables. This is a P2P-disabled SHM condition on the existing NVSwitch server, not a separate PCIe-only server. The failed original B8 attempt is excluded from results; its retry has a `_v2` label. Normal NVSwitch runs leave P2P and IB overrides unset.
