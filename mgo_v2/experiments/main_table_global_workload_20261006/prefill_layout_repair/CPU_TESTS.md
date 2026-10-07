# CPU regression evidence

`test_rank_partial_layout.py` passed28 legacy/new comparisons:7 distributions x4 ranks, including both requested full prefill sizes. Compares packet counts, IDs, expert groups and packed tensor contents to the independent legacy planner. No GPU/model timing is inferred from these synthetic checks.

Command environment uses the existing `br_ca_carep_cpu_headroom_20261003/cpu_deps` PYTHONPATH for Numba, the existing phase01 Python environment, and OMP_NUM_THREADS=1. Numba compilation was explicitly warmed before CPU timing observations. Real model validation receipts supplement these tests with48-layer/rank GPU index equality.
