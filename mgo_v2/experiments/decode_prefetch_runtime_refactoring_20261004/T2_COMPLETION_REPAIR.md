# T2 physical completion repair

The original overlap T2 path called `ForwardPacket.finish()` before prediction. That calls NCCL `Work.wait()` and enqueues unpack operations on the current CUDA stream, but does not establish host-side physical completion. The installed PyTorch `ProcessGroupNCCL.hpp` explicitly documents default `wait()` as stream synchronization; host blocking requires a timeout/blocking mode. The experiment environment did not request that mode. Thus the old T2 rows measure a stream-ordered variant, not the PLAN section 13 post-forward CPU trigger.

The repair records an event after forward completion/unpack and synchronizes that event before T2 prediction. This does not wait for the separate background H2D stream. T0/T1 and the existing barrier path are unchanged. BF16 remains fixed, with no precision comparisons.

Keep the running M13_BF16_SCREEN process and all its records. It already imported the old code, so its remaining T2 rows are historical only and ineligible for PLAN T2 selection. After that process exits, run `M13_T2_SYNC_REPAIR_CONFIG.json` with stage `M13_T2_SYNC_REPAIR`: BR only, both batches, P1/P2/P4, 64-step prefix, existing adaptive rules. New workers record `t2_host_completion=true`. Physical validation and measurements are still pending; syntax checks are not a physical PASS.

The final screen must assemble unchanged original T0/T1 rows and separately validated repaired T2 rows, preserving source paths/hashes and every original sample. Do not overwrite original raw results or combine old/new T2 samples. Only then choose full-256 finalists. Selector rejects legacy unmarked T2 rows.
