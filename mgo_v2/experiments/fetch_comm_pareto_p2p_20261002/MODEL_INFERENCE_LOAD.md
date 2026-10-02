# Resident model inference load — above 90% on eight GPUs

Owner requested higher utilization after the B4-B32 research packet completed.
The idle workload is Qwen1.5-MoE-A2.7B-Chat BF16, one resident model per GPU,
using `mgo_v2/examples/model_inference_load.py`. It performs real model forwards,
not standalone GEMM. No research measurement was rerun.

GPU-local batch is now **1024**, max128 input tokens, no retained KV cache,
last-token logits only. The saved 256-question workload cycles four times to
construct this load batch. This is independent of the research B32 setting.
Batch64 reached roughly 60% and batch256 roughly 83%; batch512 reached roughly
90% inconsistently. At batch1024, moving resource checks to a background
thread removed the observed gaps in the final validation window.

## Validation

Twenty snapshots at two-second intervals, approximately 40 seconds total:

| GPU | Min utilization | Mean utilization | Max utilization | Max temperature |
|---:|---:|---:|---:|---:|
| 0 | 95% | 96.05% | 97% | 75 C |
| 1 | 94% | 95.90% | 97% | 62 C |
| 2 | 94% | 95.45% | 97% | 60 C |
| 3 | 95% | 96.35% | 97% | 71 C |
| 4 | 92% | 94.95% | 96% | 72 C |
| 5 | 95% | 95.70% | 97% | 61 C |
| 6 | 94% | 95.60% | 97% | 71 C |
| 7 | 91% | 95.80% | 97% | 60 C |

All 160 GPU samples exceeded 90%. Every worker produced successful inference
receipts. Maximum device memory used was 37,183 MiB (36.31 GiB); minimum free
memory was 43,907 MiB (42.88 GiB). No OOM. This is an observed window, not a
promise that every future utilization sample will exceed 90%.

## Operating controls

- Live settings: `/home/hwlee/mgo-results/model_inference_load_20261003/settings.json`.
  `{"batch":1024}` is active; the code default is also 1024. Batch changes are
  read every five seconds without reloading model weights.
- Receipts, PID registry and logs stay in the same directory. The existing
  experiment launcher pauses these owned workers and resumes them afterward.
- Torch memory is capped at 65% of device capacity. Resource checks remain
  active every five seconds in a background thread; a detected stop reason
  ends the loop after the current forward.
- Exit if another compute job appears on the GPU, temperature reaches 85 C,
  GPU free memory falls below 8 GiB, host available memory falls below 128 GiB,
  a STOP file appears, or SIGTERM/SIGINT is received. Monitor exceptions also
  stop inference. Other users' jobs are never terminated.

Evidence: [utilization samples and worker receipts](model_inference_load_high_utilization.json).
The original lower-load receipt remains historical in `model_inference_load_status.json`.
