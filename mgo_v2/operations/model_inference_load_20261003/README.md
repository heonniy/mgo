# Resident model inference load

After completion of the BR/CA/CA-rep study, the owner requested efficient CPU
resource use for future experiments and resident model load on all eight GPUs.
The completed 1552 CPU cells were not repeated; dynamic refresh remains stopped.

GPU 0--7 each run `mgo_v2/examples/model_inference_load.py` with a resident
Qwen1.5-MoE-A2.7B-Chat model, bfloat16, batch 1024 and input length at most 128.
Each iteration performs a real model forward. These are operational load
workers, not scientific measurements or new experiment results.

`verification.json` records matching live PIDs, completed iterations, and six
GPU samples spaced five seconds apart. This is a point-in-time verification,
not a guarantee of future utilization or uptime.

The existing worker limits Torch allocation to 65% of VRAM and checks every
five seconds for another compute process, GPU free memory below 8 GiB,
temperature at least 85 C, host available memory below 128 GiB, or a STOP file.
A guard requests exit at the next forward boundary. Each worker uses one CPU
thread. No automatic restart overrides a guard stop.

Live settings, PID registry, telemetry and this restart's logs are under
`/home/hwlee/mgo-results/model_inference_load_20261003/`; `current_run.txt`
identifies the new log directory. Earlier registry and STOP marker were
preserved there before restart. Create the root `STOP` file to request all
workers to exit, and verify GPU memory is released before starting experiments.

Future authorized CPU studies should scale independent cells against measured
throughput, CPU affinity and memory headroom, with single-thread numerical
libraries, shared/read-only inputs where practical, and bounded memory use.
The previous study's 16-worker limit is historical. This host exposes 192 CPUs;
available resources and other users must be checked again at each launch.
Record concurrency, throughput and peak RSS rather than maximizing CPU usage
through redundant work. See the latest policy at the top of `mgo_v2/AGENTS.md`.
