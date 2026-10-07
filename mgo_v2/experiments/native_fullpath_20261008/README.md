# Using the native candidate

`--expert-executor native` selects the C++ expert executor for decode.
`--native-prefill` additionally selects it for optimized prefill. The flag is
valid only with `--prefill-optimized`. `--decode-layout-fast` selects the
previously validated compiled decode packet construction. All three are
explicit options in the low-level worker. The main-table supervisor's
`--ours-final` profile now selects the validated combination below.

The measured full-path candidate uses:

```text
--expert-executor native --native-prefill
--prefill-optimized --prefill-layout-fast --decode-layout-fast
--policy LA_CA_NEAR --prefetch-off
```

Compile the extension outside timing with the matching PyTorch/CUDA toolchain.
On this host the task-local Ninja is in
`/home/hwlee/mgo-tools/native-expert-build/bin`; set `CUDA_HOME=/usr/local/cuda`,
`TORCH_CUDA_ARCH_LIST=9.0` and `MAX_JOBS=1`. The normal supervisor handles
host/HBM guards and restores only owned inference loads.

Read [RESULTS.md](RESULTS.md) before selecting prefill native on a new
workload. B64/L512 showed a strong decode gain but no measured TTFT gain.
The B16 prefetch/placement comparisons select OFF/Near for the new main-table
OURS profile; the other cells still require their own measurements. Historical
H0 receipts and the old expanded table remain intact. The new expanded-table
runner gives native-final OURS rows a separate label and uses two unfiltered
repeats rather than the historical five-sample triplet.
The paired native BR/Near TPOT component analysis is in
[RANK_DIAGNOSIS.md](RANK_DIAGNOSIS.md).
