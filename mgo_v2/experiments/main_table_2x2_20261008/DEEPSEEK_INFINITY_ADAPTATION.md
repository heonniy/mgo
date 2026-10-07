# DeepSeek MoE-Infinity C30 adaptation

The DeepSeek-V2-Lite MoE-Infinity wrapper now calls the EAM predictor after
each routed dispatch and updates native soft eviction priorities. Under the
DeepSeek C30 budget (499 routed-expert slots total, about 125 per GPU), native
expert completion did not finish when speculative transfers were also queued:

| Smoke attempt | DeepSeek speculative candidates | Observation |
|---|---:|---|
| `mt2_deepseek_sharegpt_b16_l512_infinity_smoke_v7` | 232 through target layer 3 | blocked inside `expert_dispatcher.wait_expert()` on the second token |
| `mt2_deepseek_sharegpt_b16_l512_infinity_smoke_v8` | 56 through target layer 3 | same block, despite free-space backpressure |
| `mt2_deepseek_sharegpt_b16_l512_infinity_smoke_v9` | 0 | two-token warmup and target completed; 52 EAM calls in each phase |

The raw traces and process stacks remain under `/home/hwlee/mgo-results/headline_r4_20261007/`.
The observed association is with speculative transfer submission; the exact
native deadlock mechanism has not been proven. For the DeepSeek baseline only,
the default therefore computes EAM scores and retains their eviction
priorities while admitting no speculative transfers. Qwen's audited EAM path
is unchanged. This is a model-specific baseline limitation, not a claim that
DeepSeek has working speculative EAM prefetch. The C30 residency budget and
the three-repeat timing contract remain unchanged.

The opt-in `MGO_DEEPSEEK_EAM_MIN_FREE_SLOTS` environment variable exists for
bounded diagnostic probes. The main table uses the documented default.
