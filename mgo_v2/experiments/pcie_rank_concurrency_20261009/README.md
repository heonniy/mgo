# Pinned-host H2D PCIe concurrency study

Run after the Qwen three-baseline C20–C50 sweep passes all twelve baseline
cells. The user-selected physical GPU set is **0, 1, 3, 4, 5, 6, 7**; GPU2 is
excluded. This newer instruction overrides the earlier 0/1/4/5-only limit for
this microbenchmark. Never stop any process on GPU3/6/7, and abort safely if
any selected GPU remains occupied. Stop/restore only the existing owner model
loads on 0/1/4/5 using the established helper.

For every one of the 127 nonempty subsets, use one persistent process and
CUDA stream per selected GPU to copy from rank-private pinned host storage to
that GPU. Test 9 MiB (Qwen expert) and 16.5 MiB (DeepSeek expert) payloads,
four warmup copies, then 256 sequential asynchronous copies per repeat. Use
two complete, unfiltered repeats. Workers receive a common future start time;
report their actual launch skew. CUDA event elapsed time yields each rank's
effective GiB/s, and the earliest host launch through the latest completion
yields subset aggregate GiB/s. Both include only H2D work, with no model,
expert compute, NCCL, or prefetch. It is an *effective* host-to-device rate,
not a PCIe line-rate specification or a direct end-to-end prediction.

Store every repeat in `RAW.json` for resumption, and aggregate median plus
complete range in `RESULTS.json`. The runner checks at least 384 GiB host
memory before setup, 96 GiB throughout, 2 GiB GPU free memory, and successful
completion of the preceding baseline sweep. Device buffers total 25.5 MiB
per GPU, plus the CUDA context; each worker also pins 25.5 MiB of host
memory. The final report should compare each rank's subset
throughput against its solo throughput at the same payload size, report
cardinality and individual combinations, and flag start-skew or external
activity that limits interpretation.

Run with `python mgo_v2/scripts/run_pcie_rank_concurrency.py` from the repo
root, using the CUDA-enabled headline environment. Do not run alongside the
Qwen baseline sweep.
