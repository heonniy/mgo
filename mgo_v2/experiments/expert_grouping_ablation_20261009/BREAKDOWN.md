# R4 decode critical-path breakdown

This is a separate, instrumented full 64-token target on Qwen3-30B ShareGPT R4/C30/B16/input512. Each arm reproduces its unprofiled target tokens, H2D bytes and final cache state. The window starts when the first token is ready and ends when the last token is ready. Values below are exclusive GPU current-stream spans on the slowest diagnostic rank; each column sums to 100%.

| Component | A ms/token | A % | B ms/token | B % | C ms/token | C % |
|---|---:|---:|---:|---:|---:|---:|
| Attention, dense and output | 47.73 | 8.3% | 45.86 | 10.6% | 48.68 | 11.2% |
| Router gate | 9.54 | 1.6% | 9.59 | 2.2% | 9.44 | 2.2% |
| Routing metadata | 52.30 | 9.0% | 53.57 | 12.3% | 53.64 | 12.4% |
| Placement and index | 32.79 | 5.7% | 31.90 | 7.3% | 31.83 | 7.3% |
| Demand H2D submission | 7.68 | 1.3% | 12.99 | 3.0% | 7.35 | 1.7% |
| Forward dispatch | 59.63 | 10.3% | 71.19 | 16.4% | 60.23 | 13.9% |
| Exposed H2D wait | 0.00 | 0.0% | 79.79 | 18.4% | 55.17 | 12.7% |
| Expert execution and preparation | 244.59 | 42.3% | 25.80 | 5.9% | 54.49 | 12.6% |
| Return and combine | 69.20 | 12.0% | 56.54 | 13.0% | 65.52 | 15.1% |
| MoE runtime residual | 54.86 | 9.5% | 47.32 | 10.9% | 47.08 | 10.9% |
| **Instrumented total** | 578.33 | 100% | 434.55 | 100% | 433.44 | 100% |

The grouped GEMM kernel is included in Expert execution and preparation. Required H2D waiting is separate. Demand H2D submission is host/current-stream setup, while H2D copy-stream service runs concurrently and must not be added to the above total.

| Fine phase | A ms/token | A % | B ms/token | B % | C ms/token | C % |
|---|---:|---:|---:|---:|---:|---:|
| `attention_dense_residual` | 47.73 | 8.3% | 45.86 | 10.6% | 48.68 | 11.2% |
| `expert_grouped_gemm_kernels` | 0.00 | 0.0% | 7.52 | 1.7% | 13.24 | 3.1% |
| `forward_token_a2a_submit` | 12.56 | 2.2% | 11.76 | 2.7% | 11.93 | 2.8% |
| `layout_cpu` | 5.95 | 1.0% | 5.76 | 1.3% | 5.75 | 1.3% |
| `layout_device_materialization` | 12.92 | 2.2% | 12.54 | 2.9% | 12.47 | 2.9% |
| `metadata_cpu_parse` | 3.60 | 0.6% | 3.52 | 0.8% | 3.64 | 0.8% |
| `metadata_demand_histogram` | 1.50 | 0.3% | 1.42 | 0.3% | 1.49 | 0.3% |
| `metadata_device_to_host` | 1.47 | 0.3% | 1.49 | 0.3% | 1.36 | 0.3% |
| `metadata_gate_history` | 9.97 | 1.7% | 9.56 | 2.2% | 9.42 | 2.2% |
| `metadata_gpu_packet_pack` | 7.05 | 1.2% | 7.04 | 1.6% | 6.94 | 1.6% |
| `metadata_rank_all_gather` | 23.48 | 4.1% | 25.31 | 5.8% | 25.78 | 5.9% |
| `moe.current_controller` | 4.07 | 0.7% | 3.89 | 0.9% | 3.93 | 0.9% |
| `moe.demand_h2d` | 7.68 | 1.3% | 12.99 | 3.0% | 7.35 | 1.7% |
| `moe.expert_compute` | 244.59 | 42.3% | 18.28 | 4.2% | 41.26 | 9.5% |
| `moe.forward_a2a` | 31.14 | 5.4% | 31.00 | 7.1% | 30.45 | 7.0% |
| `moe.forward_complete` | 15.94 | 2.8% | 28.43 | 6.5% | 17.85 | 4.1% |
| `moe.metadata` | 5.24 | 0.9% | 5.21 | 1.2% | 5.01 | 1.2% |
| `moe.return_a2a` | 28.52 | 4.9% | 24.86 | 5.7% | 23.11 | 5.3% |
| `moe_other` | 54.86 | 9.5% | 47.32 | 10.9% | 47.08 | 10.9% |
| `placement_controller_cpu` | 9.85 | 1.7% | 9.70 | 2.2% | 9.68 | 2.2% |
| `required_h2d_exposed_wait` | 0.00 | 0.0% | 79.79 | 18.4% | 55.17 | 12.7% |
| `return_token_a2a` | 40.68 | 7.0% | 31.68 | 7.3% | 42.42 | 9.8% |
| `router_gate_compute` | 9.54 | 1.6% | 9.59 | 2.2% | 9.44 | 2.2% |

The metadata all-gather and token return include waiting for other ranks. Their event spans do not establish pure network transmission time. The separately instrumented total is higher than the unprofiled TPOT because the diagnostic adds event markers and tracing. Use PRIMARY_RESULTS.md for the performance comparison.
