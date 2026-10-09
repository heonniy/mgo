# Add fixed-owner Static to the previous main_OURS policy cells

Measure Static `expert_id % 4` on the earlier R4 Qwen ShareGPT
input128/32-decode cells, using the original native C++ Ready-First
individual-expert path, compiled prefill/decode indices, prefetch OFF,
and the same C30/C60 MAIN capacities. No grouped executor is attached.

Capture routes under Near, matching the original main_OURS policy-gap
workers, then replay the frozen route and teacher tokens for two clean
Static repeats after cache reset. Require per-rank route hashes to equal
the original cell's capture before adding the Static value to its row.
Keep all raw values and report the two-run mean and full range. Previous
BR/CA/Near rows each had one clean primary, so the appended Static value
does not retroactively make their gains statistically stable.

The scope is all earlier physical cells: C30 NVSwitch four, C60 NVSwitch
six, C30 env2 four and C60 env2 four. Use guarded serial jobs on GPUs
0/1/4/5. The env2 jobs use `NCCL_P2P_DISABLE=1` plus `NCCL_IB_DISABLE=1`
for the established same-host SHM path. Preserve other users' processes
and restore the owned model loads after each job.

For the quiet-host confirmation cell `ShareGPT_R4_C30_B8_L128_O33_s14_d5`
on the P2P-disabled transport, additionally measure a fixed-seed random
owner baseline with `main_OURS`. Seed 42 permutes 128 expert owners within
each layer, giving exactly 32 assigned experts per rank per layer; an
expert's owner remains fixed throughout the replay. Freeze Near's routing
and teacher tokens and require all rank hashes to match the BR/Near/Static
quiet-host pair. Keep both unfiltered cold-cache repeats and do not mix this
random owner result into the 18-cell Static sweep.
