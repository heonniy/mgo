# CPU prefill layout repair results

R4/C30, Near/H0/full-pinned, prefill probability-tail + BF16 rank-partial transport. Same inputs, cache budget, placement policy, BF16 accumulation order and decode implementation. Physical GPUs0/1/4/5 only. Two unprofiled full64-output primary repeats per cell after validation and warmup; dynamic expert state reset before every batch. No sample filtering or extra repetitions.

## Primary timings

Mean ± sample SD, n=2; seconds. These are clean timings, excluding the subsequent one-token diagnostic.

| Cell | TTFT | TPOT | E2E | TTFT samples |
|---|---:|---:|---:|---|
| B16_L256 | 2.432717 ± 1.197961 | 0.762082 ± 0.006765 | 50.443880 ± 1.624186 | 3.279803 / 1.585630 |
| B64_L512 | 5.143451 ± 1.166386 | 0.997978 ± 0.011600 | 68.016046 ± 1.897194 | 5.968210 / 4.318692 |

## Historical before/after

Same two-repeat protocol, sequential runs rather than an interleaved A/B. Old reference is the prior prefill-optimized implementation, not the original unfused baseline or owner-selected subset. Report timing spread rather than claiming stable speedup.

| Cell | Old mean TTFT | New mean TTFT | Observed reduction | New TTFT spread |
|---|---:|---:|---:|---:|
| B16_L256 | 4.230684 | 2.432717 | 42.50% | 69.64% |
| B64_L512 | 20.888599 | 5.143451 | 75.38% | 32.07% |

## Targeted diagnostic evidence

Seconds, four-rank mean service spans. CUDA intervals include host submission gaps; own-thread CPU is also preserved in RESULTS.json. Old diagnostic followed one-token warmup; new diagnostic follows full-generation primaries. The preparation difference prevents attributing unrelated component movements entirely to this repair. H2D service is not added into the additive timeline.

| Cell | Component | Before | After |
|---|---|---:|---:|
| B16_L256 | layout_cpu | 1.202698 | 0.095151 |
| B16_L256 | layout_device_materialization | 0.548090 | 0.037991 |
| B16_L256 | placement_controller_cpu | 0.159041 | 0.156138 |
| B16_L256 | moe_other | 0.086755 | 0.035270 |
| B16_L256 | moe.current_controller | 0.017720 | 0.009487 |
| B16_L256 | moe.expert_compute | 1.883986 | 0.495157 |
| B16_L256 | required_h2d_exposed_wait | 0.297090 | 0.294975 |
| B16_L256 | attention_dense_residual | 0.072842 | 0.073587 |
| B64_L512 | layout_cpu | 11.065628 | 0.689031 |
| B64_L512 | layout_device_materialization | 4.094870 | 0.182184 |
| B64_L512 | placement_controller_cpu | 1.115331 | 1.066041 |
| B64_L512 | moe_other | 0.526114 | 0.041682 |
| B64_L512 | moe.current_controller | 0.057320 | 0.011028 |
| B64_L512 | moe.expert_compute | 1.926869 | 0.532015 |
| B64_L512 | required_h2d_exposed_wait | 0.301548 | 0.296949 |
| B64_L512 | attention_dense_residual | 0.482018 | 0.482829 |

## Exactness and implementation

28 independent CPU layout/device-pack comparisons pass: empty, uneven, zero-active, variable effective lengths, single-owner and both full prefill sizes across four ranks. Numba compilation was excluded from the CPU timing observations and warmed before physical primaries. Each physical cell passes192 exact old/new GPU index comparisons (48 layers x4 ranks), including canonical peer/token/expert ordering, send/receive counts and expert row/column order. Metadata remains exact. All finite-logit/cache/no-Torch-recompilation guards pass.

| Cell | Identical generated token positions, both primaries | End-state/role/controller equality |
|---|---:|---|
| B16_L256 | 8192 | True |
| B64_L512 | 32768 | True |

No numerical-association change is introduced by this layout repair. Exact output parity is against the preceding BF16 rank-partial version; that version still differs from the original expert-order return.

The new Numba builder walks rank-contiguous tokens, produces canonical peer/token/expert packets with bounded eight-element ordering, and builds stable expert groups by counting. NumPy arrays are retained through one contiguous device transfer. Unused exact-return arrays and duplicate selected-ID copies are omitted. Legacy paths remain available; enable `--prefill-optimized --prefill-layout-fast`. The Numba helper uses the existing dependency, not a new package.

## Remaining opportunities

Placement controller still performs token-level generic substitution/accounting passes with substitution disabled; its measured residual cost is a bounded next target. Potential repairs include specialized exact no-substitution preparation, fewer token scans and avoiding unused temporary arrays while preserving slot/eviction decisions and accounting. These have been inspected but are not implemented in this change. Miscellaneous host work partly disappears with the index objects; attention/dense and required communication must not be treated as removable overhead. No claim that first-repeat TTFT variability is fixed.

All raw receipts/diagnostic segments and archive hashes are retained. Resource samples remain in the raw job directories. Owned model loads restored on0/1/4/5; no other GPUs touched.
