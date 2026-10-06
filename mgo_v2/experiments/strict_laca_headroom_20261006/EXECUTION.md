# Owner-authorized strict headroom execution

The owner authorized latest commit82b093db and all physical GPUs0..7 for R8
on this server on2026-10-06. This supersedes PLAN section8's prior allocation
restriction. R4 and L256 capture use0,1,4,5. Never terminate foreign workers.
Prior non-strict TTFT stays closed; this packet is the active priority.

Scope: the eight C30 TTFT cells from PLAN, R4/R8 x localB16/B64 x L256/L512.
Reuse the existing immutable ShareGPT_LONG512 requests and L512 capture;
recapture the last256 tokens rather than slicing existing routes. Decode
strict support is present but no additional decode matrix is queued.

Sequence: L256 capture -> eight CPU screen jobs -> eight CPU input preparation
jobs -> R4 physical cells -> R8 physical cells -> fresh-process S2 comparisons
and separate diagnostics. Seed ranges, top8 retention, and bounded repeats
remain unchanged. R8 authorization is explicit via environment and physical
GPU list. Resident model loads are paused before capture/physical timing and
restored on0,1,4,5 after use; R8 authorization does not silently expand idle loads.

Execution repairs relative to82b093db:
- Initialize controller package before legacy CPU policy imports.
- Preserve dependency path and add foreign-process, host/GPU memory and
  temperature gates, outside primary timing windows.
- Parallelize independent CPU cells/cases without changing search candidates.
- Separate the compute barrier from return-A2A CUDA timing; measure controller
  wall time separately from routing/admission/layout in diagnostic passes only.
- Save and publish each physical pair with source hashes; preserve failures.
- Add a persistent ordered queue and a repeated-result/diagnostic report.

The primary comparison is BR vs LA_CA_NEAR on identical selected input/routing.
The LA comparison uses the same triple, with its own paired BR baseline.
This remains deliberately BR-adversarial maximum headroom, not average-case.
No timing claims before S2. Warm correctness and physical-copy gates remain.
