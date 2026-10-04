# LA physical validation

Owner authorized physical BR vs LA verification and all eight GPUs (2026-10-04).
Start with the two LOAD winners from calibrated CPU result `1e2e10c`:
R8, local B128/B256, cache60 (3686 total slots), Gate W128, substitution OFF,
decode256, Env1 NVSwitch. Other workloads/environments are not in this queue.

Reuse exact requests, selected experts, routing weights, gate scores and teacher
tokens from the original trace. Native router computation remains, but routing
consumes the common frozen trace. No trace recapture. A live controller executes
metadata exchange, BR/LA placement, layout creation, dispatch, pinned H2D,
global fetch barrier, expert compute and return/combine. Controller and common
frozen-input staging costs remain inside E2E and TPOT; this is a controlled
full-model replay, not free-running text generation.

Before GPUs: prove the live BR/LA adapter's per-event max rank rows, H2D counts,
remote return rows and final slots/owners match the full CPU oracle, and prove
the oracle still matches the archived result. Both policies use the winner's
placement seed. Save immutable inputs and hashes outside Git, proofs in Git.

One workload/policy at a time. Eight 80GB GPUs, allocator cap 85%; initial free
GPU memory >76000 MiB, host reserve 768 GiB; untimed reserve 8 GiB/GPU and
256 GiB host. Stop only owned idle model workers. Abort on foreign GPU work.
Do one full untimed compile/correctness pass before clean reset MEASURE passes.
Validate every live fetch action against CPU references, all final cache states,
H2D bytes and within-policy argmax repeatability. No compilation in MEASURE.
No nvidia-smi/process-memory polling inside timing; resource scans at boundaries.

Per workload/policy: two MEASURE repeats; if either E2E/TPOT relative difference
exceeds 5%, label unstable and stop repeating. If either exceeds 2% but neither
exceeds 5%, do exactly one third; otherwise report mean/median and full range.
Use median with a third; no primary gain claim for unstable pairs. BR then LA;
report that order as a temporal-drift limitation. No open-ended repetitions.

Commit/push each measurement checkpoint and failures. Restore owned resident
model load on all eight GPUs after queue completion/failure. Preserve failure
receipts; fix implementation errors before retrying, never silently change cells.
