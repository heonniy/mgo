# Is ready-first H2D/compute overlap helping?

After the four-capacity post-cleanup timing comparison, run a diagnostic-only
C20 A/B on the same frozen ShareGPT B16/input512/output64 workload. Keep
main_OURS Near placement, native C++ expert execution, compiled layouts,
full-pinned source, and prefetch OFF. Clear expert cache after warmup.

The **only runtime intervention** is inside the native expert executor: after
the existing dispatch has completed, wait for all current-layer demand H2D
copies to finish on the host *before* choosing ready expert groups. The normal
arm keeps ready-first execution on the main stream while the copy stream
works. Do not add a global barrier, change owner placement, evict/admit a
different expert, or alter dispatch/return packet layout. This ablation
removes H2D overlap with expert execution but still permits H2D/dispatch
overlap. It also collapses ready-first waves, so the TPOT difference measures
the **net** value of overlap versus wave/orchestration overhead, not pure DMA
service alone.

Run two unfiltered clean target repeats per arm. Record per-run native wave,
group and wait counts, the ablation's host wait time, TPOT, TTFT, E2E, and H2D
bytes. Require exact request IDs, all output tokens, cache/ownership hashes,
finite logits, no recompilation, and identical copy bytes across arms. If
output/routing diverges, report the mismatch and do not interpret the timing
as a same-route causal comparison. A short smoke pass must precede the A/B.
No primary table row is replaced by this intervention.
