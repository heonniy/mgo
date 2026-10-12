# Induced H2D contention on GPUs 0-3 (calibration)

`scripts/h2d_contention_hog.py` runs as separate processes on GPUs 0-3. It copies 9 MiB experts from an
8 GiB random pinned pool to the GPU, either paced (`--gibps`) or back-to-back with k streams (`--streams`).
`calibrate.py` measures the in-run-like 8-GPU burst: FAST quota, 24 GiB random pinned pool, paced
submission. Results are in `CALIBRATION.json` (rate levels) and `CALIBRATION_streams.json` (stream count).

| Hog on GPUs 0-3 | DMA GPUs 0-3 | DMA GPUs 4-7 | A/B | burst makespan |
|---|---:|---:|---:|---:|
| none | 0.257 ms | 0.229 ms | 1.12 | 2.58 ms |
| 5 GiB/s | 0.285 | 0.231 | 1.24 | 2.85 |
| 10-30 GiB/s or max, 1 stream | 0.298-0.300 | 0.231-0.232 | 1.29-1.30 | 3.0-3.1 |
| max, 2 streams | 0.402 | 0.233 | 1.72 | 4.44 |
| max, 4 streams | 0.405 | 0.234 | 1.73 | 4.51 |
| max, 8 streams | 0.422 | 0.236 | 1.79 | 4.54 |

GPUs 4-7 are essentially unaffected. The controllable asymmetry range is 1.12x (current) to ~1.8x.
The hog also reads host memory (~140 GiB/s over four GPUs at max) and uses one CPU core per GPU.
