# Physical fetch-barrier execution mode

The GPU validation runtime now has a phase-separated mode matching the owner
requested execution order:

```
router
  -> metadata exchange
  -> rank / expert-owner decision
  -> activation dispatch
  -> miss expert H2D
  -> global fetch barrier
  -> MoE expert compute
  -> return + combine
```

## Runtime switches

Use both:

```
--execution-order fetch-barrier
--schedule-mode live
```

`fetch-barrier` is intentionally rejected for a non-PLAN physical run unless
`schedule-mode=live`. The live controller redoes metadata exchange and rank
selection every layer and checks that the decision matches the validated frozen
PLAN before dispatching.

The existing `streaming/frozen` path is unchanged and remains the baseline.

## Barrier semantics

After dispatch finishes, H2D starts. If the globally planned event has at
least one miss fetch:
1. dispatch CUDA/NCCL work is synchronized before CPU staging or H2D begins;
2. every local miss expert is fetched;
3. the pinned H2D stream (or default stream for pageable mode) is synchronized;
4. all EP ranks enter a distributed barrier;
5. expert GEMM starts only after the barrier.

If there is no global miss in the event, the H2D/barrier phase is skipped.

This is deliberately not an overlap-optimized runtime. Its purpose is to make
communication and expert-compute scheduling effects visible without H2D hiding
them.

## Nsight ranges

The worker emits:
- `moe.metadata_exchange`
- `moe.rank_decision`
- `moe.dispatch`
- `moe.h2d_fetch`
- `moe.h2d_global_barrier`
- `moe.expert_compute`
- `moe.combine`

These ranges should be used to verify that dispatch/H2D/compute do not overlap
in the new mode.

## Example

With an existing validated PLAN:

```bash
python scripts/run_env_offload_cell.py \
  --cell R --policy BR --phase MEASURE --environment env1 \
  --comm-mode current --h2d-mode pinned \
  --cache-ratio 0.60 --substitution off \
  --execution-order fetch-barrier --schedule-mode live
```

Run the CPU-only source guard before a GPU experiment:

```bash
python scripts/test_fetch_barrier_execution_order.py
```
