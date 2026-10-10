# R8 serial GEMM + inline H2D + FAST quota: Near vs random placement

Three rounds with alternating order, 6 samples per arm. Full table: `RESULTS_TABLE.md`.

| Arm | TPOT median [range], s | vs FAST+Near | per round |
|---|---:|---:|---|
| FAST+Near | 0.4264 [0.4216, 0.4310] | 0 | — |
| FAST+Random | 0.4357 [0.4309, 0.4394] | **+2.19%** | +2.27, +1.99, +2.52 |

- In serial mode Near placement beats random by ~2.2%. Every random sample is slower than the
  Near sample at the same round and repeat position (repeat 1: +2.3%, repeat 2: +2.2%).
- Together with `../serial_placement_r8_b16_20261011`, the serial TPOT cost tracks compute
  imbalance (CPU proxy critical/mean expert rows): Near 1.14 -> 0, Random 1.53 -> +2.2%,
  Worst 2.82 -> +21.8%.
- In grouped mode Near and random were indistinguishable (`../placement_stage2_r8_b16_20261011`),
  because grouped GEMM is short and overlaps H2D.
