# NVLink bandwidth ladder — blocked before measurement

**BLOCKED_PRIVILEGE (2026-10-03).** No bandwidth-mode write, GPU communication
experiment, model run, or CPU Pareto rerun was performed.

The read-only `nvidia-smi nvlink -gBwMode` query returned code **4** and
`Unable to read nvlink bandwidth mode`. The installed NVIDIA-SMI manual
maps code 4 to insufficient permission to access the device or perform the
operation. The current account is UID 1003 (`hwlee`). This is a query-level
permission failure; hardware support and the current FULL state remain
unverified. The CLI advertises bandwidth get/set syntax, and its installed
manual lists FULL/OFF. No supported device mode is inferred from that alone.

All eight GPUs had zero compute processes, 0 MiB used and 0% utilization
in the pre-change snapshot. The idle gate passed; the mandatory current-FULL
gate could not be verified. Per `NVLINK_BW_LADDER.md`, stop without sudo,
password prompts, job termination, or a mode write.

| Condition | L0 calibration | L1 trace | Transport receipt |
|:---|:---|:---|:---|
| FULL-P2P | NOT_RUN | NOT_RUN | No preflight launched |
| OFF-P2P | NOT_RUN | NOT_RUN | No preflight launched |
| SHM | NOT_RUN | NOT_RUN | No preflight launched |

No T_FULL/T_OFF/T_SHM values or ratios exist. This result cannot establish
any ladder ordering. CPU Pareto files and frozen schedule metadata match
their prior SHA256 hashes and the owner plan commit. F/K/C remain unchanged.

Restoration and a post-restore snapshot are **not applicable** because OFF
was never requested and no mode write occurred. A post-gate all-GPU snapshot
is included instead; it is not evidence that bandwidth mode is FULL.
All eight GPUs remained at 0 MiB with zero compute processes.

Stop here for owner review. No retries, transport tuning or model work.

Evidence: [result JSON](nvlink_bw_ladder.json), [CSV](nvlink_bw_ladder.csv),
[validation](nvlink_bw_ladder_validation.json),
[pre-change snapshot](nvlink_bw_ladder_pre_change.json),
[post-gate snapshot](nvlink_bw_ladder_post_gate.json), and
[installed CLI documentation](nvlink_bw_ladder_cli_evidence.txt).
