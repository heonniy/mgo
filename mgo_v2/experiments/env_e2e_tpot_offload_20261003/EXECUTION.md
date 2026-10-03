# Physical execution conventions

Source plan: cf6e00f. R8 and R4 transport preflights passed for both environments;
resident model load workers were stopped and all eight GPUs released memory.

The experiment-only runtime loads dense Qwen3 weights on GPU and keeps all
expert weights in a read-only memory-mapped CPU pool, fully faulted in before execution. Only explicitly allocated cache slots hold
GPU expert copies. Each miss copies all three BF16 expert matrices (9 MiB).
Activation dispatch and weighted expert returns use actual NCCL all-to-all.
Expert matrix multiplications and activation execute on GPU. This avoids the
legacy executor's per-event instrumentation and its prohibition on admitting
an unused replica. No production controller or historical study is changed.

PLAN uses the live model's routes, weights and Gate W128 history, with the same
incremental policy semantics as the completed CPU replay (validated on actual
inputs). BR seed is 42. CA-rep's strictly future raw-demand table is frozen from
the exact master trace, with the same request ordering and 256 horizon as the
CPU study. It is an exact-baseline demand oracle, not an oracle for a new
substitution-induced trajectory; any such distinction must accompany results.
PLAN stores each policy's actual generated tokens, actions and final cache.
No exact-baseline output match is required for substitution ON.

Requests are the first R*B manifest requests, assigned i%R, with original prompt
IDs, SDPA, EOS suppressed and 256 generated tokens. As in master capture, one
prefill generates g1 and 256 decode forwards consume g1..g256; final logits are
discarded. TPOT denominator is the plan's 256 decode forwards.

MEASURE uses frozen indices/actions and live routing weights; a device-only
route mismatch flag is checked after synchronization at the end. It performs
no controller/Hungarian/oracle work, route dumps, per-event wall/CUDA timing,
profiling or hit/miss counter collection. Cache-key updates are the necessary
physical residency state, not instrumentation. Only outer and decode boundary
timing is used. Compilation must be absent after warmup. COUNTERS is a separate
process that independently recomputes live policy counters, verifies its actions
against the frozen schedule, and counts actual transfers separately.

COMPILE uses a dedicated Env/cell/policy Inductor/Triton directory. Every phase
is a fresh process. An untimed full-schedule warmup precedes each MEASURE and
logical/physical residency, request position and RNG reset before timing.
Expert buffer contents may remain allocated, but every initial slot is empty
and must incur the scheduled H2D again. No expert cache hit survives reset.

Start with all GPUs empty, >=768 GiB host available and GPU temperatures <65 C.
The phase launcher checks at five-second intervals: >=256 GiB host available,
aggregate proportional set size (PSS) <=768 GiB, GPU free >=8 GiB and temperature <85 C;
foreign GPU work or a STOP file ends the phase. Torch allocation cap is 85%.
One phase has a two-hour safety timeout. No scientific samples are accepted
from a failed phase. Only the declared R/B64 resource failure permits B32.

## Resource preflight correction

The initial P/BR PLAN attempt exceeded the conservative 768-GiB aggregate RSS
guard during pool preparation and was stopped before generation/timing. No OOM
was observed. The initial per-rank pinned copy was replaced by shared file-backed
CPU expert pages; every page is touched outside timing and residency is checked
before each generation. Pageable-source transfer staging is included in real
H2D execution cost. Matrix axes, thresholds and resource limits are unchanged.
Merged substitution weights accumulate in float32 then round once to BF16,
matching the CPU policy mass semantics. The failed attempt is retained.

The shared-pool attempt exposed aggregate RSS double counting across shared
file mappings and compiler subprocesses: 861 GiB summed RSS while host available
remained 1421 GiB. The guard now counts aggregate PSS (shared pages counted
proportionally) at the same 768-GiB cap, and retains the independent 256-GiB
host-available guard. Raw RSS and PSS are both recorded. The earlier RSS-guard
failures are preserved, not treated as OOM or accepted scientific samples.
A separate initialization failure from missing generation_config was corrected
by loading the checkpoint generation_config explicitly.
