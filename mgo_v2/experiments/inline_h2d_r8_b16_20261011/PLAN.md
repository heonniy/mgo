# R8 grouped decode: per-copy staging-thread submission vs inline one-pass submission

Same cell and protocol as `../pcie_split_grouped_r8_b16_20261010` (grouped hit-then-miss,
prefetch OFF, frozen Near teacher tokens). The only change is `--inline-demand-h2d`
(commit 4a90f71): each layer's demand copies are submitted in one pass on the
calling thread, not one by one by the staging thread.

Arms: NEAR and FAST, each with and without inline submission. Two rounds with
rotated order, two target repeats per job: four samples per arm. Same-session
controls (no reuse of the 10/10 rows). NEAR (staging thread) is the baseline.
