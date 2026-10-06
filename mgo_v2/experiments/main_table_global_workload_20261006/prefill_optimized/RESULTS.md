# Prefill tail metadata + fused transport results

Both requested R4/C30 Near/H0/full-pinned cells completed on GPUs0/1/4/5. Input manifests, local/global batch, input lengths and output64 match prior measurements. No other policy, CPU/GPU allocation or H2D overlap change. Two measured repeats per cell after a separate numerical/metadata validation and full64-token warmup; reset expert/history state before measured batches.

## Primary measurements

Seconds; mean ± sample standard deviation (n=2). All samples retained; unstable means any metric range/mean exceeds5%.

| Cell | TTFT | TPOT | E2E | Timing |
|---|---:|---:|---:|---|
| B16/L256 | 4.230684 ± 1.187570 | 0.751621 ± 0.000816 | 51.582777 ± 1.238976 | UNSTABLE |
| B64/L512 | 20.888599 ± 1.097247 | 1.030342 ± 0.003001 | 85.800152 ± 0.908212 | UNSTABLE |

## Every measured sample

| Cell | Repeat | TTFT | TPOT | E2E |
|---|---:|---:|---:|---:|
| B16/L256 | 1 | 5.070422 | 0.752198 | 52.458866 |
| B16/L256 | 2 | 3.390945 | 0.751044 | 50.706689 |
| B64/L512 | 1 | 21.664470 | 1.028220 | 86.442355 |
| B64/L512 | 2 | 20.112728 | 1.032464 | 85.157950 |

## Historical comparison

B16 reference is the mean of all three original final-confirmation samples; B64 reference is the subsequent owner-requested single-shot rerun. This is sequential historical comparison, not interleaved causal validation. Additional diagnostic prefill precedes the new warmup. Do not substitute selected historical samples, claim a stable speedup, or attribute the whole difference to either individual change.

| Cell | Historical TTFT | New mean TTFT | Observed reduction |
|---|---:|---:|---:|
| B16/L256 | 4.617002 | 4.230684 | 8.37% |
| B64/L512 | 24.240570 | 20.888599 | 13.83% |

## Validation and numerical semantics

All192 layer/rank metadata comparisons per cell agree exactly with full gather, and all primary cache/copy consistency, finite-logit and no-compilation guards pass. CPU distributed regression passes7 cases including empty/unequal ranks. Default runtime selection regression passes. Initial CPU test invocation omitted the existing numba dependency path and failed before tests; rerun with the harness dependency path passed.

Fused BF16 rank-partial accumulation changes association; it does not preserve bitwise expert-order outputs. Layer numerical comparisons use identical expert contributions, not an independent end-to-end baseline. Native autoregressive output differences versus the stated baseline repeat1 are reported below; they are not an accuracy evaluation and affect subsequent decode routes.

| Cell | Max layer relative L2 | Max abs | First-token differences | All output differences | New repeats differ |
|---|---:|---:|---:|---:|---:|
| B16/L256 | 0.4986% | 8.000000 | 3/64 | 1472/4096 | 0 |
| B64/L512 | 0.4605% | 4.000000 | 10/256 | 6131/16384 | 0 |

## Implementation limits

Only probability history payload is reduced to the exact global last128 rows. Selected expert IDs and routing weights still cover every token, as do CPU policy/layout work. The CPU probability array shrinks from64MiB to64KiB per layer/rank at B64 and8MiB to64KiB at B16. These are analytical array sizes, not measured wire bytes: padded all-gather still transmits up to128 rows per rank. Prefill uses packed forward plus BF16 rank-partial return; required-H2D wait/global barrier remains and prefill overlap stays disabled. This is not grouped-GEMM fusion. No default promotion or extra model runs.

All receipts/logs are archived with hashes in RESULTS.json. Full resource series remain in the recorded raw job directories. Supervisors pass and restore owned idle model loads only on GPUs0/1/4/5.
