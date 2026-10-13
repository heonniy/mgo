# EP cost breakdown: BR vs LA_CA_NEAR vs CA, and local batch 16/32/64 (Qwen3, R4, C30, serial)

Qwen3-30B-A3B, R4 (physical GPUs 0/1/4/5), ShareGPT, input 512, output 64, C30.

**Main findings**
- **LA_CA_NEAR lowers EP cost at every batch.** It cuts compute imbalance by 17–40% and token-row imbalance from 1.27–1.31 to 1.03–1.04 (max/mean over ranks).
- **CA and CA_NATIVE reduce communication volume the most and help nothing.** They cut remote rows by 18% and bytes by 9%, yet communication time and EP cost stay at BR's level. Their compute imbalance is unchanged from BR, and their TPOT is the worst of all policies.
- **Communication time does not follow bytes.** Over every run, remote bytes span 59–260 MiB per step while comm cost stays at 134–148 ms per step.
- **The TPOT gain of LA_CA_NEAR over BR shows only at B16 (−1.8%).** At B32 and B64 it is within run-to-run noise (−0.3% / −0.1%, n=2).

## Setting

- **Serial ablation mode:**
  - native serial per-expert executor;
  - `--h2d-serial-ablation` (no H2D/compute overlap);
  - prefetch off;
  - BR prefill, with the decode policy switched at decode step 1.
- **Jobs:** each job runs one uninstrumented target, whose TPOT is reported, then a separate post-generation diagnostic pass, which gives the breakdown. There are 2 jobs per cell; the order was BR, NEAR, NEAR, BR. CA and CA_NATIVE ran afterwards.
- **Workloads:**
  - B16: `qwen_cache_ablation_20261009` C30 (same as the 6-repeat main study).
  - B64: `main_table_2x2_20261008` ShareGPT.
  - B32: derived from B64, taking the first 32 rank-local requests per rank (`manifests/`). B16 is not a rank-local subset of B64, so request sets differ slightly across batches.
- **Policies:**
  - **BR:** balanced quota, random placement.
  - **LA_CA_NEAR:** compute balance first; locality only among near-balanced ranks, with δ = 2%.
  - **CA:** code 1. Hungarian maximization of local token demand within the balanced quota; no compute balance.
  - **CA_NATIVE:** same objective as CA, solved by the C++ min-cost-flow kernel.
  - **BR_SALT46:** BR with a different random stream (code 26, salt 46).
- **Contamination check:**
  - CA and CA_NATIVE ran with a GPU watcher (`*_gpu_watch.log`); no foreign process appeared.
  - The B32/B64 worst-salt runs in `contaminated_by_other_session/` overlapped another session's jobs on the same GPUs. They are excluded.

## Results (means over 2 runs; ms per decode step unless noted)

| cell | n | TPOT | ep_cost | load_imbalance | compute_imbalance | h2d_wait_imbalance | comm_cost | comm_imbalance | controller | experts_ratio | rows_ratio | bytes_ratio | remote_rows_per_layer | remote_MiB_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B16/BR | 2 | 0.5704 | 179.6 | 37.8 | 34.6 | 3.3 | 141.8 | 52.1 | 9.1 | 1.137 | 1.308 | 1.005 | 385.3 | 65.4 |
| B16/BR_SALT46 | 2 | 0.5699 | 180.3 | 35.9 | 29.9 | 6.1 | 144.3 | 63.7 | 9.3 | 1.136 | 1.286 | 1.008 | 384.7 | 65.5 |
| B16/CA | 2 | 0.5732 | 179.4 | 37.5 | 33.3 | 4.2 | 141.9 | 51.3 | 17.4 | 1.135 | 1.308 | 1.020 | 316.6 | 59.3 |
| B16/CA_NATIVE | 2 | 0.5747 | 182.7 | 34.4 | 29.1 | 5.3 | 148.3 | 54.9 | 13.5 | 1.132 | 1.302 | 1.015 | 316.5 | 59.2 |
| B16/LA_CA_NEAR | 2 | 0.5599 | 168.3 | 30.0 | 21.5 | 8.5 | 138.3 | 48.4 | 10.0 | 1.086 | 1.043 | 1.008 | 341.3 | 63.4 |
| B32/BR | 2 | 0.6455 | 182.9 | 38.6 | 32.6 | 6.0 | 144.4 | 54.7 | 10.7 | 1.115 | 1.268 | 1.003 | 768.5 | 130.8 |
| B32/LA_CA_NEAR | 2 | 0.6426 | 174.0 | 29.4 | 19.5 | 9.9 | 144.6 | 54.6 | 11.7 | 1.070 | 1.030 | 1.005 | 695.0 | 127.7 |
| B64/BR | 2 | 0.7048 | 205.6 | 63.5 | 52.0 | 11.5 | 142.0 | 64.5 | 12.8 | 1.108 | 1.313 | 1.003 | 1536.5 | 260.2 |
| B64/LA_CA_NEAR | 2 | 0.7036 | 186.8 | 52.0 | 43.0 | 9.0 | 134.8 | 54.9 | 13.9 | 1.075 | 1.038 | 1.000 | 1433.5 | 258.2 |

### B16: BR vs LA_CA_NEAR vs CA

| | BR | LA_CA_NEAR | CA | CA_NATIVE |
|---|---|---|---|---|
| remote rows / layer | 385 | 341 (−11%) | **317 (−18%)** | 317 (−18%) |
| remote MiB / step | 65.4 | 63.4 | **59.3 (−9%)** | 59.2 |
| compute imbalance | 34.6 | **21.5** | 33.3 | 29.1 |
| rows imbalance (max/mean) | 1.308 | **1.043** | 1.308 | 1.302 |
| comm cost | 141.8 | **138.3** | 141.9 | 148.3 |
| EP cost | 179.6 | **168.3** | 179.4 | 182.7 |
| controller | 9.1 | 10.0 | 17.4 | 13.5 |
| TPOT (s/token) | 0.5704 | **0.5599** | 0.5732 | 0.5747 |

Even with the controller moved to C++ (CA_NATIVE), communication-only placement does not lower EP cost or TPOT.

**Why:**
- **Compute.** Per-layer compute is set by the rank that executes the most experts, at about 0.18–0.19 ms per expert.
  - The busiest-compute rank is the rank with the most experts in 88–92% of layers.
  - CA leaves that count as unbalanced as BR (experts ratio 1.135 vs 1.137).
- **Communication.** Comm time does not shrink with fewer bytes (slope ≈ 0 across all runs).

## Files

- **Per-run receipts in `raw/` and `../ep_breakdown_r4_serial_20261013/raw/`:** small JSONs, logs, and `layer_breakdown.npz`, which holds the per-layer, per-rank arrays used here. It was extracted with `extract_npz.py`.
- **Full ~50 MB diagnostic JSONs:** kept outside git, in `/home/hwlee/mgo-results/ep_breakdown_{r4_serial,batch}_20261013/`.
- **Analysis:** `python analyze.py` rebuilds `SUMMARY.json` and the tables.
- **`trace_b{32,64}_BR/`:** decode demand and gate traces, used by `trace_salt.py` for the offline worst-salt search (`salt_search_b*.jsonl`).
- **`run*.sh`:** exact drivers.
