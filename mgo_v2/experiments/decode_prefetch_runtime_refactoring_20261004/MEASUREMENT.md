# Measurement and phase attribution

The final result must explain not only TPOT but **why** TPOT changed.

## 1. Layer measurement window

For decode, define one MoE-layer span from the start of global routing metadata
exchange to completion of local combine.

Record CPU and GPU intervals separately.

## 2. Required intervals

### CPU
- metadata pack;
- metadata collective submission/wait;
- current-layer controller;
- predictor calculation;
- prefetch placement controller;
- packet packing / layout preparation.

### GPU / transport
- metadata NCCL;
- forward A2A;
- demand H2D;
- prefetch H2D;
- expert GEMM / expert-kernel range;
- return A2A;
- combine kernels.

Asynchronous H2D duration must come from CUDA/CUPTI intervals, not Python
enqueue wall time.

## 3. H2D exposure

Let `H` be the union of all CPU->GPU expert-copy intervals for the current
layer's demanded experts plus background next-layer prefetch copies. Let

```
U = forward_A2A union expert_compute union return_A2A
```

Then report:

```
H2D_hidden = |H intersect U|
H2D_exposed = |H| - H2D_hidden
H2D_hidden_ratio = H2D_hidden / |H|
```

Also split H2D bytes/time into:
- current mandatory demand;
- useful prefetch;
- wasted prefetch;
- prefetch that was still inflight at use.

## 4. Exclusive / overlap decomposition

Report the interval union table per layer and decode aggregate:

```
H2D only
COMM only
COMPUTE only
H2D ∩ COMM
H2D ∩ COMPUTE
COMM ∩ COMPUTE
H2D ∩ COMM ∩ COMPUTE
idle / unattributed inside MoE span
```

Where:

```
COMM = metadata NCCL union forward A2A union return A2A
COMPUTE = expert-kernel intervals
```

This avoids double-counting phase sums.

## 5. Controller exposure

Current-layer controller is on the critical path before urgent H2D can start.
Report its full wall time.

The next-layer predictor/prefetch controller is intended to run after forward
A2A launch. Report:

```
prefetch_controller_total
prefetch_controller_hidden_by_A2A
prefetch_controller_exposed
```

The optimized design passes only if predictor/controller work does not erase
the H2D benefit.

## 6. Prefetch metrics

For every P and trigger position:

- predictions issued;
- precision;
- recall;
- actual-miss recall;
- ready-at-use hits;
- inflight-at-use hits;
- queued-at-use hits;
- wrong predictions;
- useful prefetch bytes;
- wasted prefetch bytes;
- canceled-before-H2D count;
- promotion count;
- promotion victim count;
- promotion-induced future reloads;
- residual demand H2D;
- exposed demand H2D.

## 7. Communication metrics

Verify physically:
- one forward A2A per layer;
- one return A2A per layer;
- no extra payload A2A;
- metadata collective count separately.

Report:
- forward wire bytes;
- return wire bytes;
- active peer degree;
- median/p90 forward time;
- median/p90 return time;
- alpha-dominated fraction from small-message calibration if available.

## 8. Main result table

For B128 and B256:

| Policy | P/rank | PF trigger | PF ready | PF inflight | Wasted PF | Demand H2D | H2D exposed | H2D hidden % | Fwd A2A | Expert | Return A2A | Current ctrl | PF ctrl exposed | TPOT |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Rows:
- BR no-prefetch;
- BR prefetch;
- CA prefetch;
- LA prefetch.

## 9. Nsight validation

At least one final B128 and B256 run must include NVTX ranges:
- `moe.metadata`;
- `moe.current_controller`;
- `moe.demand_h2d`;
- `moe.forward_a2a`;
- `moe.prefetch_controller`;
- `moe.prefetch_h2d`;
- `moe.expert_compute`;
- `moe.return_a2a`;
- `moe.combine`.

The trace must visually confirm overlap and the expected two payload A2As.
