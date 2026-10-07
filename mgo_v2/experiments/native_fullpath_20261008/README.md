# Using the native candidate

`--expert-executor native` selects the C++ expert executor for decode.
`--native-prefill` additionally selects it for optimized prefill. The flag is
valid only with `--prefill-optimized`. `--decode-layout-fast` selects the
previously validated compiled decode packet construction. All three are
explicit options; the headline default remains H0 and legacy decode layout.

The measured full-path candidate uses:

```text
--expert-executor native --native-prefill
--prefill-optimized --prefill-layout-fast --decode-layout-fast
```

Compile the extension outside timing with the matching PyTorch/CUDA toolchain.
On this host the task-local Ninja is in
`/home/hwlee/mgo-tools/native-expert-build/bin`; set `CUDA_HOME=/usr/local/cuda`,
`TORCH_CUDA_ARCH_LIST=9.0` and `MAX_JOBS=1`. The normal supervisor handles
host/HBM guards and restores only owned inference loads.

Read [RESULTS.md](RESULTS.md) before selecting prefill native on a new
workload. B64/L512 showed a strong decode gain but no measured TTFT gain.
The two-repeat, frozen-routing comparisons do not replace the main table.
