# Cache-size sweep ablation (C20 / C30 / C40)

OURS (HAQ) and RANDOM fetch-rank placement in serial ablation mode. The existing baseline sweeps on the
same frozen requests are shown alongside for reference. Models: Qwen3-30B-A3B-Instruct-2507 and
DeepSeek-V2-Lite-Chat.

**Main result:** both policies' TPOT falls monotonically as the cache grows, on both models.
RANDOM is 10–12% slower than OURS on Qwen3 and 14–16% slower on DeepSeek, at every cache size.

## Setting

- **Workload:** R4 on physical GPUs 0/1/4/5 (NVLink), ShareGPT, rank-local B16 (64 global requests),
  input 512, output 64, greedy, no EOS stop.
- **Manifests:** frozen `qwen_cache_ablation_20261009/WORKLOADS.json` and
  `deepseek_cache_ablation_20261009/WORKLOADS.json`. Every cache size uses identical requests.
- **Serial ablation mode (OURS and RANDOM):**
  - native serial per-expert executor;
  - `--h2d-serial-ablation`: the host waits for all of a layer's demand H2D before expert compute;
  - prefetch off;
  - prefill placement BR; the decode policy switches at decode step 1;
  - each target starts from an empty expert cache.
- **Samples:** every target is kept. Qwen3 has 2 rounds × 3 targets = 6 per cell, in the order
  (C30, C20, C40) then reversed. DeepSeek has 2 jobs × 2 targets = 4 per cell.

## Policies

| Name | Code | Rule |
|---|---|---|
| OURS (HAQ) | 16 | Hit-aware quota (stage 2). Water-fills each rank's miss quota against its resident hits. |
| RANDOM | 19 | Each miss independently picks a uniform random rank. No quota, so a rank can receive 0 copies or all of them, e.g. [4,0,0,0]. |

RANDOM was added in commit 3943785.
- The numba RNG is reseeded every event from `MGO_RANDOM_SALT*1000003+event`, so all ranks agree on the placement.
- The salt is the round number (1 or 2). All targets within one job share one random draw.

## Results

### Qwen3: serial ablation, OURS vs RANDOM (TPOT s/token, median [min, max], n)

| Cache | OURS (HAQ) | RANDOM | RANDOM vs OURS |
|---:|---|---|---:|
| C20 | 0.5822 [0.5798, 0.5897] (6) | 0.6492 [0.6467, 0.6520] (6) | +11.5% |
| C30 | 0.5510 [0.5486, 0.5564] (6) | 0.6139 [0.6086, 0.6178] (6) | +11.4% |
| C40 | 0.5255 [0.5204, 0.5306] (6) | 0.5791 [0.5770, 0.5817] (6) | +10.2% |

### Qwen3: all systems (medians)

| System | C20 TPOT | C30 TPOT | C40 TPOT | C20 TTFT / E2E | C30 TTFT / E2E | C40 TTFT / E2E |
|---|---:|---:|---:|---:|---:|---:|
| OURS (HAQ, serial) | 0.5822 | 0.5510 | 0.5255 | 1.75 / 38.6 | 1.75 / 36.7 | 1.75 / 35.1 |
| RANDOM (serial) | 0.6492 | 0.6139 | 0.5791 | 1.75 / 42.7 | 1.74 / 40.6 | 1.74 / 38.3 |
| OURS (default overlap mode) | 0.6448 | 0.5621 | 0.5379 | 2.10 / 42.7 | 2.05 / 37.2 | 2.02 / 38.1 |
| llama.cpp balanced | 0.5253 | 0.4806 | 0.4355 | 161.94 / 195.0 | 145.07 / 175.3 | 129.53 / 157.0 |
| MoE-Infinity (repaired) | 3.1430 | 3.2272 | 3.2043 | 5.21 / 203.2 | 5.19 / 208.5 | 5.55 / 207.4 |
| DeepSpeed ZeRO-3 offload | 4.0453 | 4.1219 | 4.1117 | 5.89 / 260.7 | 5.57 / 266.0 | 5.95 / 265.0 |

### DeepSeekV2Lite: serial ablation, OURS vs RANDOM (TPOT s/token, median [min, max], n)

| Cache | OURS (HAQ) | RANDOM | RANDOM vs OURS |
|---:|---|---|---:|
| C20 | 0.3458 [0.3437, 0.3475] (4) | 0.3994 [0.3990, 0.4009] (4) | +15.5% |
| C30 | 0.3353 [0.3349, 0.3378] (4) | 0.3838 [0.3813, 0.3847] (4) | +14.5% |
| C40 | 0.3206 [0.3170, 0.3229] (4) | 0.3641 [0.3617, 0.3650] (4) | +13.6% |

### DeepSeekV2Lite: all systems (medians)

| System | C20 TPOT | C30 TPOT | C40 TPOT | C20 TTFT / E2E | C30 TTFT / E2E | C40 TTFT / E2E |
|---|---:|---:|---:|---:|---:|---:|
| OURS (HAQ, serial) | 0.3458 | 0.3353 | 0.3206 | 0.72 / 22.5 | 0.59 / 21.7 | 0.72 / 20.9 |
| RANDOM (serial) | 0.3994 | 0.3838 | 0.3641 | 0.73 / 25.9 | 0.75 / 24.9 | 0.73 / 23.7 |
| OURS (default overlap mode) | 0.2879 | 0.2847 | 0.2817 | 0.57 / 18.7 | 0.60 / 18.6 | 0.59 / 18.6 |
| llama.cpp balanced | 0.4134 | 0.4135 | 0.3485 | 145.21 / 171.3 | 145.38 / 171.5 | 118.48 / 140.4 |
| MoE-Infinity (repaired) | 1.2598 | 1.2591 | 1.2169 | 1.62 / 81.1 | 1.67 / 81.0 | 1.64 / 78.3 |
| DeepSpeed ZeRO-3 offload | 4.7116 | 4.6384 | 4.6506 | 4.85 / 302.0 | 4.91 / 297.0 | 4.80 / 297.8 |

## Interpretation

Over a whole token, RANDOM's per-rank copy totals are nearly even. For Qwen3 C30, target 2:

| Policy | Copies/token by rank |
|---|---|
| RANDOM | [694, 683, 689, 685] |
| HAQ | [691, 690, 688, 686] |

The cost comes from imbalance within each layer. In serial mode every layer waits for its busiest
rank, so per-layer skew adds up to the 10–16% TPOT gap.

## Baseline rows

Baseline rows are copied unchanged from the existing sweeps:
- Qwen3: `../qwen_cache_ablation_20261009/BASELINE_SWEEP_RESULTS.json` and `RESULTS.csv`
- DeepSeek: `../deepseek_cache_ablation_20261009/RESULTS.csv`

Notes on each baseline, from a code audit on 2026-10-10:

- **OURS (default overlap mode)** is the earlier main-table configuration: grouped GEMM with overlap.
  It is not the serial mode used above.
- **llama.cpp balanced:**
  - Expert placement is static and whole-layer. llama.cpp stores each layer's 128 experts as one tensor,
    and the run uses an equal layer count per GPU. Effective residency is therefore about 83% of the
    budget: C20 16.7%, C30 25.0%, C40 33.3% of experts.
  - Non-resident experts are computed on the CPU (`op_offload=false`). This favours its TPOT and
    explains the 120–160 s TTFT.
  - The 4 GPUs run as a layer pipeline over one 64-request batch.
- **MoE-Infinity (repaired):**
  - At the owner's request, these were repaired: EAM, priority eviction, per-GPU budgets, and
    multi-GPU fixes. It is not unmodified upstream.
  - Per-layer host-side Python prefetch work dominates TPOT, so TPOT is flat in cache size.
  - Speculative admission is uncapped on Qwen3. It is capped on DeepSeek only, as a workaround for a
    native dispatcher stall.
  - It runs one 64-request batch with expert-parallel homes.
- **DeepSpeed ZeRO-3 offload:**
  - There is no expert cache. Each MoE block is gathered and released on every token.
  - The cache-% setting only scales the live-parameter and prefetch window, so a flat curve is
    expected by design.

## Caveats

- **Tokens differ between policies, so routes diverge.** Each HAQ cell has a single argmax_hash.
  Each RANDOM cell has two, one per salt, because the rank-partial BF16 combine order follows placement.
  This is an end-to-end comparison on the same prompts, not a fixed-route one.
- **C30 OURS is reused.** `raw/qwen_c30_HAQ_r*` are copies of `../haq_serial_decode_20261010/raw/c30_HAQ_r*`.
  The DeepSeek C30 HAQ samples are the HAQ targets of `raw/ds_haq_serial_c30_v1` and
  `raw/ds_static_c30_v1`, copied from `../haq_serial_deepseek_20261010/raw`. Same manifests and command.
- **One job ran uncommitted code.** `raw/qwen_c20_HAQ_r2` started after the RANDOM edit but before
  its commit. Its status.json records c9c00af, but it ran c9c00af plus the RANDOM diff. That diff only
  adds a new branch and assert exemptions; the HAQ path is unchanged. All later jobs record 3943785.

## Reproduce

- `run_ours.sh` and `run_random.sh` are the exact job drivers as run. Their output root was
  `/home/hwlee/mgo-results/haq_cache_ablation_20261010`.
- `python summarize.py` rebuilds `SUMMARY.json` and these tables from `raw/` and the baseline sweeps.
