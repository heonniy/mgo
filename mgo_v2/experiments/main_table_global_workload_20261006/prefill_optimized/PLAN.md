# Owner-requested prefill metadata + fused transport experiment

R4/C30, Near/H0/full-pinned, GPUs0/1/4/5 only. B64/L512 first, then B16/L256. Same frozen ShareGPT inputs,64 output tokens and expert budgets as the completed main-table packet.

Opt-in prefill changes: retain exact global last128 probability rows for Gate history instead of full probabilities; keep complete selected IDs and routing weights for policy decisions. Enable fused forward packet and BF16 rank-partial return in prefill. Preserve existing prefill H2D barrier and no-overlap behavior, decode path and default runtime.

Validation: seven CPU distributed cases (empty/uneven and both experiment sizes), then a separate one-token full-model validation on warmup inputs. Check metadata against original gather for all48 layers, and compare rank-partial output to exact expert-order return using identical expert contributions at every layer. Record BF16 numerical differences. Do not claim end-to-end token identity from local comparisons.

Then reset, run one disjoint full64-token warmup and two clean primaries, resetting expert/cache/history state before each. No outlier removal or automatic extra repetitions. Compare with all retained historical attempts; sequential before/after is not a causal decomposition. Report instability explicitly. No diagnosis timers in primary.

Supervisor guards: host start384GiB, abort96GiB; exclusive target GPUs; restore only owned idle model workers. Preserve every receipt/failure.
