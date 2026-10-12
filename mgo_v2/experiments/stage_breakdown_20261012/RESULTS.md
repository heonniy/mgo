# Stage-wise cost breakdown (serial default environment), Qwen3 and DeepSeek R8

Scope cut at the owner's request: 3 repetitions per cell, not the planned 4. The runner stopped on the
STOP file after the 30th job, so STATUS.json says FAIL (AssertionError on STOP). All 30 jobs passed.
Figure: `stage_breakdown.png` / `.pdf` (`figure.py`). Numbers: `SUMMARY.json` (`analyze.py`).
Values are decode sums in seconds, mean ± std over 3 runs.

## (1) Stage 1 quota: NEAR -> FAST

| Model | Arm | H2D straggler | H2D imbalance | TPOT | vs NEAR |
|---|---|---:|---:|---:|---:|
| Qwen3 | NEAR | 3.92 ± 0.01 | 0.65 ± 0.01 | 0.4333 | 0 |
| Qwen3 | FAST | 3.65 ± 0.06 (-7%) | 0.44 ± 0.01 (**-33%**) | 0.4303 | -0.7% |
| DeepSeek | NEAR | 3.16 ± 0.02 | 0.55 ± 0.00 | 0.2621 | 0 |
| DeepSeek | FAST | 2.88 ± 0.00 (-9%) | 0.27 ± 0.01 (**-51%**) | 0.2572 | -1.9% |

## (2) Stage 2 placement on the FAST quota (plus HAQ_FAST)

| Model | Arm | compute imbalance | comm cost | EP cost | EP vs FAST+Near | TPOT | vs FAST+Near |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 | FAST+Worst | 6.47 ± 0.01 | 10.86 ± 0.07 | 17.47 ± 0.11 | +85% | 0.5245 | +21.9% |
| Qwen3 | FAST+Random | 2.20 ± 0.00 | 8.45 ± 0.04 | 10.79 ± 0.03 | +14% | 0.4407 | +2.4% |
| Qwen3 | FAST+Near | 1.31 ± 0.04 | 7.96 ± 0.13 | 9.43 ± 0.19 | 0 | 0.4303 | 0 |
| Qwen3 | HAQ_FAST | **1.05 ± 0.02** | **7.58 ± 0.09** | **8.67 ± 0.12** | **-8%** | 0.4264 | -0.9% |
| DeepSeek | FAST+Worst | 1.73 ± 0.03 | 4.35 ± 0.11 | 6.14 ± 0.14 | +28% | 0.2791 | +8.5% |
| DeepSeek | FAST+Random | 0.90 ± 0.02 | 4.03 ± 0.06 | 5.02 ± 0.07 | +4% | 0.2610 | +1.5% |
| DeepSeek | FAST+Near | 0.71 ± 0.01 | 3.99 ± 0.01 | 4.81 ± 0.01 | 0 | 0.2572 | 0 |
| DeepSeek | HAQ_FAST | **0.61 ± 0.02** | 3.96 ± 0.02 | **4.64 ± 0.04** | **-4%** | 0.2529 | -1.7% |

HAQ_FAST keeps the stage-1 H2D benefit: H2D imbalance Qwen 0.46 (FAST 0.44, NEAR 0.65), DeepSeek
0.32 (FAST 0.27, NEAR 0.55).

## TPOT ladder versus NEAR

| Step | Qwen3 | DeepSeek |
|---|---:|---:|
| NEAR -> FAST (quota / H2D) | -0.7% | -1.9% |
| FAST -> HAQ_FAST (placement / EP) | -0.9% | -1.7% |
| NEAR -> HAQ_FAST | **-1.6%** | **-3.5%** |
| FAST+Worst (bad placement) | +21.1% | +6.5% |

Notes:
- Qwen TPOT is the uninstrumented timed target. DeepSeek TPOT is the instrumented teacher target,
  so its absolute value is inflated, but arms are comparable.
- Qwen diagnostics use live tokens (post-generation pass); DeepSeek uses the frozen NEAR teacher.
- HAQ_FAST's copy-cost constants (250/230 us) were calibrated on Qwen 9 MiB experts and reused for DeepSeek.
- The breakdown is span-based: comm cost is net of the wait for the busiest rank. Read the relative
  changes, not exact wire time.
