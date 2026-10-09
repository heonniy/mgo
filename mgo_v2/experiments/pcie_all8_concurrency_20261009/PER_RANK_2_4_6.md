# Per-GPU effective H2D for 2, 4, and 6 simultaneous GPUs

Every one of the 28 two-GPU, 70 four-GPU, and 28 six-GPU subsets is included at both expert payload sizes in `PER_RANK_2_4_6.xlsx` and `PER_RANK_2_4_6.csv`. Blank GPU cells mean that GPU was inactive.

The rates below are medians of CUDA-event rank-local GiB/s across the subsets in each row. Percentages are median losses against that same physical GPU measured alone at the same payload size. Each subset has two unfiltered repeats.

| Simultaneous GPUs | From GPUs 0–3 | Subsets | Qwen 0–3 GiB/s (loss) | Qwen 4–7 GiB/s (loss) | DeepSeek 0–3 GiB/s (loss) | DeepSeek 4–7 GiB/s (loss) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 6 | — | 50.30 (0.2%) | — | 50.75 (0.2%) |
| 2 | 1 | 16 | 50.34 (0.1%) | 50.39 (0.0%) | 50.80 (0.1%) | 50.83 (0.0%) |
| 2 | 2 | 6 | 49.95 (0.9%) | — | 50.48 (0.7%) | — |
| 4 | 0 | 1 | — | 46.13 (8.5%) | — | 46.51 (8.5%) |
| 4 | 1 | 16 | 50.13 (0.5%) | 49.58 (1.6%) | 50.65 (0.4%) | 50.01 (1.6%) |
| 4 | 2 | 36 | 49.71 (1.4%) | 50.25 (0.3%) | 50.25 (1.2%) | 50.70 (0.3%) |
| 4 | 3 | 16 | 43.65 (13.3%) | 50.32 (0.2%) | 43.91 (13.6%) | 50.75 (0.2%) |
| 4 | 4 | 1 | 34.63 (31.2%) | — | 34.84 (31.5%) | — |
| 6 | 2 | 6 | 48.36 (4.0%) | 45.87 (9.0%) | 48.81 (4.0%) | 46.28 (9.0%) |
| 6 | 3 | 16 | 43.52 (13.6%) | 48.95 (2.8%) | 44.04 (13.4%) | 49.41 (2.8%) |
| 6 | 4 | 6 | 34.79 (30.9%) | 49.98 (0.8%) | 35.00 (31.2%) | 50.47 (0.7%) |

Loss is 100 × (1 − subset speed / solo speed); negative values in the full table are small measured improvements and are preserved.
This is repeated pinned-host H2D copy service, not model inference speed or an isolated PCIe component measurement.

## Concrete solo → concurrent examples

The numbers below are **per GPU** effective GiB/s, not the aggregate. They
show why the number of active GPUs alone does not predict an individual GPU's
rate. The complete 2-, 4-, and 6-GPU subset/rank values remain in
`PER_RANK_2_4_6.xlsx` and `PER_RANK_2_4_6.csv`.

| Active GPUs | Qwen expert: selected GPU solo → concurrent | DeepSeek expert: same GPU solo → concurrent |
|---|---|---|
| 2, GPUs 2·7 | GPU2 50.38 → 49.46; GPU7 50.35 → 50.28 | GPU2 50.84 → 50.84; GPU7 50.83 → 50.84 |
| 4, GPUs 0·1·4·5 | GPU0 50.40 → 49.67; GPU1 50.42 → 49.48; GPU4 50.36 → 50.24; GPU5 50.43 → 50.21 | GPU0 50.86 → 50.26; GPU1 50.82 → 49.99; GPU4 50.84 → 50.72; GPU5 50.84 → 50.64 |
| 4, GPUs 0·1·2·4 | GPU0 50.40 → 43.37; GPU1 50.42 → 43.27; GPU2 50.38 → 43.42; GPU4 50.36 → 50.33 | GPU0 50.86 → 43.82; GPU1 50.82 → 43.73; GPU2 50.84 → 43.88; GPU4 50.84 → 50.77 |
| 4, GPUs 0·1·2·3 | GPU0 50.40 → 34.63; GPU1 50.42 → 34.60; GPU2 50.38 → 34.70; GPU3 49.53 → 34.63 | GPU0 50.86 → 34.82; GPU1 50.82 → 34.81; GPU2 50.84 → 34.95; GPU3 50.86 → 34.86 |
| 4, GPUs 4·5·6·7 | GPU4 50.36 → 46.17; GPU5 50.43 → 46.11; GPU6 50.42 → 46.15; GPU7 50.35 → 45.96 | GPU4 50.84 → 46.47; GPU5 50.84 → 46.48; GPU6 50.85 → 46.54; GPU7 50.83 → 46.53 |
| 6, GPUs 0·1·4·5·6·7 | GPU0 50.40 → 48.34; GPU1 50.42 → 48.06; GPUs4–7 50.35–50.43 → 45.80–45.99 | GPU0 50.86 → 48.77; GPU1 50.82 → 48.40; GPUs4–7 50.83–50.85 → 46.13–46.34 |
| 6, GPUs 0·1·2·4·5·6 | GPUs0–2 50.38–50.42 → 43.22–43.44; GPUs4–6 50.36–50.43 → 48.93–49.24 | GPUs0–2 50.82–50.86 → 43.13–43.39; GPUs4–6 50.84–50.85 → 48.86–49.06 |
| 6, GPUs 0·1·2·3·4·5 | GPUs0–3 49.53–50.42 → 34.68–34.78; GPUs4–5 50.36–50.43 → 49.86–49.96 | GPUs0–3 50.82–50.86 → 34.96–35.11; GPUs4–5 50.84 → 50.25–50.39 |

At two GPUs, the largest measured rank loss was only 1.8% for Qwen and
2.2% for DeepSeek across all 28 pairs. At four GPUs the combinations differ
sharply: the mixed 0·1·4·5 set remains close to solo speed, whereas adding
GPU2 to make 0·1·2·4 drops the 0–2 workers to about 43 GiB/s, and running
all 0–3 drops them to about 35 GiB/s. At six GPUs, four active workers from
0–3 again reach about 35 GiB/s even while two 4–7 workers remain near
50 GiB/s. This is a measured subset association, not proof of which PCIe,
NUMA, IOMMU, or pinned-memory resource is limiting.
