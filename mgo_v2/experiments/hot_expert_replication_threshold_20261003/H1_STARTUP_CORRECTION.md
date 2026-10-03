# H1 startup correction before any timing

Both original 32-KiB transport smokes passed (T0 P2P/IPC and R3 SHM).
The first timing worker failed its exact environment assertion before NCCL
initialization and before collecting any samples: repository bootstrap added
`NCCL_P2P_DISABLE=0`. The new worker omitted the post-bootstrap removal used
by both existing stable smoke and latency workers. This is an implementation
startup error, not a measured transport failure or OOM.

Preserve H1_startup_failure.json and its external raw receipts. All four model
workers were restored by the failure cleanup. Correct the worker by removing
that bootstrap default before Torch/NCCL initialization, matching existing
stable workers. Reuse the two successful smokes; run only the five unexecuted
measurement processes (four ordered pair passes and one H2D process). Keep
all sample counts, transport settings, payloads and thresholds unchanged.
The corrected first cell uses a new directory; never overwrite failure data.
