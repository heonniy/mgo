# Native baseline follow-up after the complete expansion

The owner requests all baselines after the localB16/B32/B64 and output64/128/256
best-input expansion completes, comparing against the GR-improved OURS setting.
GR here identifies the validated G-NEAR configuration: group-balanced PCIe quotas,
grouped expert GEMM, native C++ metadata/history/controller, native prefill and
the physically verified global forward→H2D→compute→return serialization.

Run one frozen cell: Qwen3-30B-A3B-Instruct-2507 BF16,
R4_C30_B16_L512_O64, ShareGPT input512, local16/global64, output64/no EOS
shortening, GPUs0,1,4,5. Use the original headline64 target/warmup manifests:
target SHA256 dea64853847b1ba1d4f3e0c8cc112eb8ba84dcde5f1d04fb90adb6fca91d67d6;
warmup SHA256 cf2aeb54933d729f55b8169ca9f0a5a5f951ce773d9d9d63490cfe4105e752bd.
The selected maximum-gain samples remain a separate workload. No model/dataset/
batch/length sweep is added to the baselines.

Measure DeepSpeed native ZeRO-Inference CPU parameter offload with48 conditional
MoE blocks marked as native ZeRO leaves; the recorded repaired Qwen3 MoE-Infinity
native pipeline; and synchronous BF16 llama.cpp balanced3 with32 physical CPU
threads, CUDA graphs/reuse OFF, GPU attention/KV and12 static GPU expert layers.
Each baseline receives a bounded progress/numerics/resource smoke, a full64 warmup
and FIVE unfiltered full64 target repetitions. Native baseline schedules remain
unchanged; do not imply their H2D/communication overlap equals serialized OURS.

Validate finite outputs, fixed source IDs and endpoints, exactly64 output tokens,
GPU KV, actual C30 residency bounds, model/code/binary hashes, host headroom128GiB,
per-device HBM, ROOT-worker plus child-process RSS and actual pinned telemetry
where available. llama's unknown incidental staging pinning is labelled unknown.
The CPU parameter-offload budget includes all parameters; static llama leaves307
expert slots unused. Preserve these native implementation differences.

The old DeepSpeed continuation failed BEFORE GPU launch because it looked for
live_cohort_validation.json while the archived file was losslessly compressed
as live_cohort_validation.json.gz. Both forms are now supported. The earlier
failure receipt remains under data2; it is not a failed inference measurement.

The new continuation freezes the expansion supervisor PID/start-ticks/argv.
It starts no baseline or CPU build until that supervisor has terminated PASS,
all eight new matrix cells are validated, the matrix is complete, and the final
result push is verified. Waits never launch on a dead RUNNING or failed predecessor.
CPU build/conversion/import steps acquire the same exclusive GPU-work lock and
set CUDA_VISIBLE_DEVICES='' while the idle burn continues. Builds do not run
alongside any primary. Only actual GPU jobs take a burn-stopping GPU lease.

Use Conda /data2/esjung/envs/mgo-pcie and CUDA12.1 toolkit. Private sources/models/
conversion scratch remain under/data2/esjung. Infinity is built with the recorded
native pinned-allocator API, CUTLASSv3.5.1 and BF16 kernels; unnecessary Blackwell
FP4 is disabled. llama is built at its frozen commit and the local frozen HF
model is converted to GGUF, verifying all144 expert tensors are actual BF16
and54GiB total, before any physical llama run.

Order after expansion: DeepSpeed preflight/smoke/full64; Infinity native
build/smoke/full64; llama native build/BF16 conversion/smoke/full64. Preserve
all failed attempts; diagnose before retrying, without reducing requested
batch/length/budget or substituting systems. Commit each completed native
baseline separately, update the G-NEAR/main comparison CSV/TeX/report, and push
after workers exit. The existing validated seven-policy OURS cohort is used as
reference; explicitly disclose its earlier measurement session and retain its
three repeats rather than relabeling them as contemporaneous five-repeat data.

```sh
/data2/esjung/envs/mgo-pcie/bin/python -u mgo_v2/scripts/continue_pcie_baselines.py \
  --predecessor /data2/esjung/mgo-results/pcie_topology_ablation_20261009/maxgain_expansion_attempt1 \
  --predecessor-pid 92308 --predecessor-start-ticks <frozen-process-identity> \
  --out /data2/esjung/mgo-results/pcie_topology_ablation_20261009/baselines_after_expansion_attempt1
```

Queued/prepared is not a baseline measurement. Build, smoke and full64 outcomes
are recorded independently; the final comparison requires all three systems PASS.
