# ShareGPT C60 policy-gap physical results

R4 on GPUs 0,1,4,5; C60 MAIN3678 plus eight reserved P2 slots; native C++ expert execution, compiled prefill/decode metadata, full pinned CPU source, BF16 rank partials, prefetch OFF. Each case has 128 input tokens, 32 decode forwards, and one clean primary per policy after warmup. Routes and teacher inputs are frozen across BR, CA_NATIVE, and Near.

| Local batch | Seed (sample/order) | Policy | Clean TPOT s/token | Diagnostic MoE s/token | Decode peer GiB | Decode H2D GiB |
|---:|---|---|---:|---:|---:|---:|
| 8 | 20/7 | BR | 0.432621 | 0.447359 | 1.026 | 204.346 |
| 8 | 20/7 | CA_NATIVE | 0.437821 | 0.463323 | 0.878 | 205.910 |
| 8 | 20/7 | Near | 0.423135 | 0.456660 | 0.996 | 203.871 |
| 8 | 14/5 | BR | 0.418036 | 0.450962 | 1.024 | 160.497 |
| 8 | 14/5 | CA_NATIVE | 0.432362 | 0.456172 | 0.894 | 162.343 |
| 8 | 14/5 | Near | 0.408085 | 0.426641 | 0.990 | 159.917 |
| 16 | 22/3 | BR | 0.505627 | 0.527861 | 2.050 | 370.222 |
| 16 | 22/3 | CA_NATIVE | 0.511198 | 0.543979 | 1.804 | 371.259 |
| 16 | 22/3 | Near | 0.501507 | 0.528586 | 1.966 | 370.090 |
| 16 | 16/3 | BR | 0.478414 | 0.501281 | 2.048 | 320.757 |
| 16 | 16/3 | CA_NATIVE | 0.501038 | 0.525695 | 1.817 | 323.209 |
| 16 | 16/3 | Near | 0.476265 | 0.504360 | 1.976 | 320.581 |
| 64 | 13/0 | BR | 0.572153 | 0.602975 | 8.215 | 558.510 |
| 64 | 13/0 | CA_NATIVE | 0.587160 | 0.618947 | 7.406 | 559.354 |
| 64 | 13/0 | Near | 0.579188 | 0.602070 | 8.009 | 558.976 |
| 64 | 20/5 | BR | 0.566874 | 0.598486 | 8.219 | 543.771 |
| 64 | 20/5 | CA_NATIVE | 0.580988 | 0.615785 | 7.438 | 545.177 |
| 64 | 20/5 | Near | 0.569150 | 0.596008 | 8.010 | 544.263 |

Best observed candidate gain versus its paired BR among the two physical seeds per batch (negative means the policy remained slower):

| Local batch | CA_NATIVE seed / gain | Near seed / gain |
|---:|---|---|
| 8 | haregpt_b8_case0 / -1.20% | haregpt_b8_case1 / +2.38% |
| 16 | haregpt_b16_case0 / -1.10% | haregpt_b16_case0 / +0.81% |
| 64 | haregpt_b64_case1 / -2.49% | haregpt_b64_case1 / -0.40% |

The MoE value is from a separate instrumented run: for each decode step, sum the 48 non-attention MLP spans per rank, take the maximum rank, then average 32 steps. It includes routing, metadata, H2D dependencies, dispatch, expert work, return and combine; it is not pure kernel time and cannot be subtracted from clean TPOT. The instrumentation can change policy ordering, so the clean TPOT decides observed performance. Peer volume counts token dispatch and return payload. H2D service can overlap other work.

Seed search screened 24 sample seeds × 8 rank-order seeds with a historical variable-context locality proxy, then physically measured the top two different sample seeds per batch on fresh input128 runs. These are bounded observed winners, not global maxima. Each policy has one clean timing sample per seed; jitter significance is unmeasured. CA_NATIVE reduced peer bytes in all six cases but was slower than BR in all six. Near was faster in B8 and B16 and slower in B64; the B16 differences are under 1%. BF16 output-token agreement with BR is recorded in SHAREGPT_RESULTS.json.

Raw run receipts are under `/home/hwlee/mgo-results/headline_r4_20261007/policy_gap_sharegpt_*_20261008/`; per-run status and trace hashes are in SHAREGPT_PROVENANCE.json. The first run began just before the harness commit; its status records the prior HEAD, while the launched worker bytes match git revision `6b8c9f4` as documented by the worker SHA-256. LMSYS-Chat-1M remains pending official gated access.
