# Async metadata candidate

Opt-in `async_metadata_inputs=true`; default remains false. Preserve the same
fixed metadata record and one collective. Header and frozen CPU Gate inputs
use persistent pinned buffers and nonblocking copies. The blocking receive
copy at the end of each serial collect completes previous input DMA before
host buffers are reused. Dynamic GPU Gate computation is unchanged.

Motivation: R4_B256_METADATA_API_DIAGNOSTIC.json identifies 6144 pre-collective
stream synchronizations across 3072 decode layers (16.308s rank0 diagnostic
API wall time). This includes prior GPU work; it is not predicted saved time
or proof of the cause of primary jitter.

4-GPU B128/B256 full-record byte parity passed with alternating old/new call
order, frozen/dynamic Gate and repeated buffer reuse. Full-model validation
precedes paired timing in R4_H64_ASYNC_METADATA. R4 GPUs0,1,4,5; decode64;
BF16; same unique-combine V3/P2/T2/torch staging. Both BR and LA enable the
same option. Two stable pairs stop, at most three. No pooling with earlier
core fingerprints; this candidate also includes f5511d6 scheduler lock fix.
No production default is activated by this experiment.
