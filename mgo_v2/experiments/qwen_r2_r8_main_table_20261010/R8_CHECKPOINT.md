# Completed Qwen R8 checkpoint

ShareGPT, input512, 64 output tokens, B16 per GPU (128 requests total), C30.
The center is the mean of two unfiltered target repeats and brackets show
their full range. TPOT includes attention; TPS is global output tokens/E2E.

| System | TTFT (s) | TPOT (s/token) | E2E (s) | TPS |
|---|---:|---:|---:|---:|
| main_OURS | 2.472 [1.816, 3.127] | 0.387 [0.382, 0.391] | 26.829 [25.882, 27.775] | 305.723 [294.936, 316.510] |
| DeepSpeed ZeRO-Inference | 6.502 [5.918, 7.085] | 4.534 [4.524, 4.545] | 292.175 [290.933, 293.417] | 28.038 [27.919, 28.158] |
| MoE-Infinity repaired | 8.736 [7.925, 9.546] | 3.629 [3.628, 3.630] | 237.363 [236.486, 238.241] | 34.513 [34.385, 34.641] |
| llama.cpp synchronous balanced1 | 308.743 [308.708, 308.777] | 0.855 [0.855, 0.856] | 362.633 [362.554, 362.711] | 22.590 [22.585, 22.595] |

main_OURS TPOT differs by 2.39% and E2E by 7.06% between its two targets;
mark its E2E pair unstable and retain both samples. The other three systems
have <1% TPOT and E2E pair differences. TTFT variation remains visible in
the ranges. The C30 llama.cpp placement audit confirms one complete expert
layer on each GPU, with 9.0 GiB resident experts under the 16.2 GiB global
expert budget. MoE-Infinity uses EAM eviction priorities with speculative
transfer admission disabled; both targets completed with zero admitted
candidates and C30 budget proofs. The repair and excluded diagnostic
attempts are documented in [INFINITY_STALL.md](INFINITY_STALL.md).

Raw receipts are outside Git under
`/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs/`:
`ours_full_v2`, `deepspeed_full_v2`, `infinity_full_v4`, and
`llama_full_v1`. The R2 main table is measured separately and uses 32 total
requests, so these R8 TPS values are not an equal-batch scaling comparison.
