# All-eight pinned-host H2D concurrency study

The owner clarified that the PCIe test must include **every physical GPU 0–7**.
Measure all 255 nonempty subsets, including GPU2, at 9 MiB and 16.5 MiB
per copy, with two unfiltered repeats. This supersedes the GPU selection in
the prior seven-GPU study, whose results remain separate and unchanged.

The runner uses one persistent worker, rank-private pinned host source, and
one CUDA stream per GPU. For each subset it performs four warmup copies, then
256 sequential asynchronous copies under a common future start time.
Results include each GPU's event-timed effective GiB/s, aggregate effective
GiB/s from earliest submission through latest completion, and launch skew.
This is effective H2D service under concurrent load, not PCIe line rate or
model inference latency.

Run only after the Qwen baseline sweep has passed. The runner stops and
restores **owned** model loads on GPUs 0/1/4/5; it never terminates foreign
processes. Abort if any selected GPU has an external compute process. Check
at least 384 GiB available host memory before launching, 96 GiB throughout,
and 2 GiB free per GPU. Device buffers and pinned host buffers are each
25.5 MiB per GPU, plus normal CUDA context memory.

From the repository root, use the CUDA-enabled headline Python environment:

```bash
python mgo_v2/scripts/run_pcie_rank_concurrency.py --gpus 0 1 2 3 4 5 6 7 --output mgo_v2/experiments/pcie_all8_concurrency_20261009
python mgo_v2/scripts/report_pcie_rank_concurrency.py --input mgo_v2/experiments/pcie_all8_concurrency_20261009
```

`RAW.json` records each repeat for resumption; `RESULTS.json`, the two CSVs,
the report, and PNG/PDF figure summarize every subset.

For per-GPU speeds at every 2-, 4-, and 6-GPU combination, see
`PER_RANK_2_4_6.xlsx` or `PER_RANK_2_4_6.csv`; the matching compact summary is
`PER_RANK_2_4_6.md`. Regenerate these with
`python mgo_v2/scripts/report_pcie_246_rank.py`.
