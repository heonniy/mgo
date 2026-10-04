# Owner 0.1% fetch/H2D tolerance amendment

Preserve the completed exact-match search as history. Reuse its249 candidate
receipts /1992 BR-CA pairs without recapture or rerunning the CPU search.
Eligibility is abs(CA-BR)/BR <=0.001 separately for total fetches and H2D bytes.
Integer arithmetic abs(CA-BR)*1000 <=BR makes the inclusive boundary exact.
All original Peer-best/Critical-best ranking and tie-break rules remain.

There are88 eligible R4 pairs and1 eligible R8 pair. R4 Peer-best is
sample251/DP42/BR99; R4 Critical-best is sample415/DP8/BR42. R8 has one common
winner sample484/DP13/BR73, so deduplicate its two objectives. There are three
unique workloads,12 initial timed generations, at most12 confirmation
measurements. Keep the0.1% mismatch explicit; do not label it exact matching.

Offline PLAN uses only the original prefill and first64 exact decode steps.
Require route/Gate hashes, final cache state, peer bytes, H2D and per-event
critical traffic to reproduce the selected CPU receipt. Both policies share
identical request/rank order, teacher tokens, top8 expert IDs and BF16 routing
weights. Return layout includes the original source token/expert weight for
each partial. Independent CPU exchange fixtures validate this alignment for
R4/R8 and unequal prefill lengths before GPU execution.

Physical runtime uses real dense layers, real expert kernels, CPU expert H2D,
bounded cache and NCCL all-to-all. Live router compute runs for compute cost
but cannot alter frozen routing. A3 sends activations, routing weights, and
weighted expert outputs. A2 sends activations and unweighted outputs, then
multiplies each returned partial by its source's frozen BF16 weight before
combine. A2 never transmits a weight tensor.

Use one resident torchrun group per R with all selected schedules loaded.
Compile artifacts persist per R. Before first timed use of each R/runtime,
run one8-step readiness prefix. Required full64 COUNTERS/correctness passes
are separate untimed work: verify physical transfer counts against CPU;
compare A3/A2 weighted returned partials for the first decode step/all48 layers;
require identical full64 teacher-forced logit-argmax hashes for each policy.
These passes also validate compiler coverage for the selected schedules.
No full schedule warmup is repeated before MEASURE. Cache/KV/RNG are reset.
MEASURE forbids recompilation and detailed counters and uses common frozen
inputs. If it recompiles or fails parity, invalidate it; no silent rerun.

Env2 only, SHM/direct/direct preflight, NCCL_CUMEM_ENABLE=0,
NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1. R4 GPUs0,1,4,6 completes first;
then R8 GPUs0..7. Each rank keeps the same disjoint24 guest vCPUs. No host-NUMA
claim. No scientific GPU overlap or resident load during the screen.
Boundary safety scans finish before GO; while a timed sample runs the parent
polls only completion files and process exit, never PSS/smaps/GPU/ps scans.
Reserve >=768GiB host and >76000MiB/GPU before launch; maintain >=256GiB host,
>8192MiB/GPU and <85C outside timing. Bound each timed generation to30min and
each R group to4h. Kill/reap only owned science processes on failure.

For each unique workload: BR/A3 then CA/A3, BR/A2 then CA/A2. After an initial
BR/CA pair, if CA has >=1% positive gain in any of E2E, decode wall or TPOT,
run BR and CA once more for that runtime. Never add a second confirmation.
Report every raw observation without confidence intervals. These are
exploratory optimized stress screens, not publication-quality repetitions,
dataset averages, or autoregressive quality measurements.

Commit/push selection, plan validation, transport, per-R results and final
handoff. Restore all eight real-model inference workers after completion or
failure. No expanded search, other batch/cache/GPU set, or automatic followup.
