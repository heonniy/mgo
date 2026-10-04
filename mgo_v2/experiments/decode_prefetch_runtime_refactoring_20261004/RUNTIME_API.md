# Selected optimized runtime

The optimized runtime exposes one default entry point and three explicit
ablation modes. All use `DecodeOffloadRuntime`, the same C+P arena/controller,
BF16 rank-partial return, and two decode payload collectives per layer.

```python
from mgo_v2 import create_selected_runtime, selected_options

options = selected_options()
runtime = create_selected_runtime(args, model, backing, experts)
runtime.stage_frozen_inputs(horizon)
# Use the existing physical harness generation/reset interface.
# Close the owned staging worker when the runtime is no longer needed.
runtime.close()
```

This is the physical harness integration interface. `args`, `model`, `backing`
and `experts` are initialized by the existing offload worker, including frozen
input paths, rank, placement policy, cache capacities and placement seed. The
caller keeps policy selection (BR/LA/CA); the runtime entry point sets the
measured execution configuration. It does not load another model or create a
second cache authority.

The default is read from this packet's `FINAL_RUNTIME_SELECTION.json`.
Until that artifact exists with a completed, frozen selection, the default
entry point fails explicitly. A provisional timing candidate does not activate
a default. No final winner is claimed by the presence of this API or document.

To run an ablation after selection, pass `arm=` to either function:

| Arm | Prefetch slots/rank | Residual-fetch barrier | Ready-first overlap |
|---|---:|---|---|
| `V1_OPT_NOPF_BARRIER` | 0 | On | Off |
| `V2_OPT_PF_BARRIER` | Frozen P | On | Off |
| `V3_OPT_PF_OVERLAP` | Same frozen P | Off | On |

V2 and V3 share the BR-only frozen trigger and prefetch budget. All modes use
BF16 and preserve the existing prefill path. The internal `physical_prefetch`
flag selects the common optimized execution path; V1 has P=0 and issues no
prefetch reservations despite that internal flag being enabled.

The physical paired worker constructs explicit arms through
`create_explicit_runtime`, which shares configuration validation and runtime
construction with the default entry point. M14 smoke and M15 physical evidence
must pass before the final selected default is published. The configuration
unit test alone is not a physical correctness or performance result.

An explicit `selection_path=` can locate a copied final selection artifact.
Its schema requires `status: PASS`, `frozen: true`, `precision: bf16`, a
`chosen_arm` from the table, and the frozen `prefetch: {P, trigger}`. Preserve
the accompanying physical evidence when moving that artifact.
