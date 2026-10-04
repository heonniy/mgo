# Coalesced return numerical validation

One prefill plus eight frozen decode steps, all eight GPUs. Numbers compare final logits and argmax against the legacy BF16 expert-order reference. Teacher inputs and router selections/weights stay frozen. These are not free-generation quality results.

| Batch | Policy | Accumulation | Argmax agreement | Different / total | Max relative L2 | Max absolute logits delta |
|---|---|---|---:|---:|---:|---:|
| 128 | BR | bf16 | 99.8264% | 16 / 9216 | 0.008949 | 0.750000 |
| 128 | LA | bf16 | 99.8264% | 16 / 9216 | 0.009256 | 1.000000 |
| 128 | BR | fp32 | 99.8806% | 11 / 9216 | 0.009063 | 1.000000 |
| 128 | LA | fp32 | 99.8806% | 11 / 9216 | 0.009063 | 1.000000 |
| 128 | BR | fp64 | 99.8806% | 11 / 9216 | 0.009063 | 1.000000 |
| 128 | LA | fp64 | 99.8806% | 11 / 9216 | 0.009063 | 1.000000 |
| 256 | BR | bf16 | 99.8318% | 31 / 18432 | 0.008320 | 0.750000 |
| 256 | LA | bf16 | 99.8481% | 28 / 18432 | 0.009004 | 1.000000 |
| 256 | BR | fp32 | 99.8264% | 32 / 18432 | 0.008793 | 1.000000 |
| 256 | LA | fp32 | 99.8264% | 32 / 18432 | 0.008793 | 1.000000 |
| 256 | BR | fp64 | 99.8264% | 32 / 18432 | 0.008660 | 1.000000 |
| 256 | LA | fp64 | 99.8264% | 32 / 18432 | 0.008660 | 1.000000 |

Every candidate preserves the CPU cache/role/copy plan and uses exactly two payload A2As per decode layer. The return contains one partial vector per received token/rank packet.

Use FP32 accumulation as the common three-arm stack. In the independent transport tests it matched the canonical high-precision contribution sum, while costing half the FP64 return bytes. BF16 remains a separately reported candidate; it is not silently excluded because of rounding differences. No placement-policy performance measurements were used for this precision choice.
