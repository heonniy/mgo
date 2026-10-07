# Current main-table BR synchronization audit

Owner requests R4/C30 BR local B16 and B64, both input256, output64.
GPUs0/1/4/5 only; shared-host 384/96GiB guards. Near/H0/full-pinned
main-table substrate with policy BR, existing prefill optimizations and original
decode layout. No compiled decode promotion or strict barrier changes.
One clean primary per cell, preceded by disjoint warmup and empty-cache reset.
Then one separate full-generation diagnostic per cell using CUDA events and
CPU clocks without extra phase barriers. Match tokens and final cache roles.
Report single-shot timings without a stability claim. Diagnostic stream spans
include CPU submission gaps and peer waits, not pure kernel/network times.
H2D service is non-additive. Preserve all raw outputs outside git and archive
summaries plus primary receipts. No additional matrix or policy comparison.
