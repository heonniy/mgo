# Balanced3 validation result

R4/C30/localB32/input512/output64, CPU32/32, fixed affinity, both graph options OFF. Warmup excluded. Actual loader ownership PASS: logicalCUDA0/1/2/3 each has3 expert layers, physical0/1/4/5. Static expert13.5GiB globally,3.375GiB/device.36 expert layers compute on CPU.

|Metric|Balanced3 mean ± sample SD|Legacy14-layer mean|Change|
|---|---|---|---|
|TTFT|290.246563 ± 0.028289|281.391786|+3.147%|
|TPOT|0.779878 ± 0.000684|0.744603|+4.737%|
|E2E|339.378847 ± 0.071376|328.301765|+3.374%|

Two full primaries pass finite-logit, placement, GPU-KV and TPOT-recomputation guards. Raw receipts attached. Observed repeat ranges below0.13% across all metrics.

Token comparison against legacy, per repeat: [{'total': 8192, 'different': 1720}, {'total': 8192, 'different': 1720}]. Repeated balanced3 tokens identical: True.

This changes both placement and GPU-resident expert layer count14->12 (CPU34->36). Slower timing does not isolate a harmful effect of balancing; reduced GPU residency is a confound. Different CPU/GPU arithmetic paths may change tokens; no quality equivalence conclusion. New native runner includes placement instrumentation. No main-table replacement or further cell expansion authorized.
