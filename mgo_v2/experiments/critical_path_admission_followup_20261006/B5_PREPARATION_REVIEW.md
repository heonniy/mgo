# B5 preparation review

Reviewed source: `528da2d6f22698cd85899b7ed7ecde0b748ee1c5`.

No B5 GPU experiment was launched. The plan explicitly says this commit does
not authorize a GPU run; matrix.json also sets execution_authorized=false and
stop_before_gpu_run=true. Existing owned model loads on GPUs 0,1,4,5 were left
running. GPUs 2,3,6,7 were not modified.

## CPU validation

With CUDA_VISIBLE_DEVICES empty:
- The committed barrier ordering/plumbing test passed.
- Both existing scheduler helper unit tests passed.
- Mocked runtime construction confirmed BR/FCA propagate the flag, omission
  defaults to false, and integer/string/null flags are rejected.
- All four changed runtime/worker Python files parse successfully.

The environment has no pytest installation; the test function was invoked
directly and scheduler tests used unittest. An initial invocation omitted the
examples/scripts import paths and failed before scheduler tests; the complete
project path passed. These are CPU checks, not physical correctness evidence.

## Execution wiring to address before a GPU launch

The committed B5 cases describe H1b BR/FCA at C30, local B128, decode64, but
are not a dedicated B5 launcher:
- run_b3_measure.py constructs its own cases without the barrier flag.
- b3_measure_worker.py always measures H0 as well as its candidate, even when
  given the B5 case file. It should not be used unchanged for H1b-only timing.
- The profile worker selects graph preparation via b3_executor, not the
  executor field in the committed B5 cases. A B5 profile launcher must map
  H1b explicitly; otherwise passing the case file directly does not select it.
- Preserve validation receipts before assertions, and compare physical copies
  and bytes with the existing canonical reference. Merely matching a newly
  generated reference can hide the prior timing-dependent cancellation.

The intended bounded run is correctness/copy parity first, then separate
Nsight diagnosis; clean primary timing must use the existing two-repeat rule.
Every full64 rank must record 3072 post-expert barriers. A copy mismatch stops
timing, without relaxing parity or retrying for a favorable sample.

Barrier duration includes its own collective service cost; it is not by itself
a measurement of rank imbalance. No C60, R8, H2 or Stage C expansion is implied.
