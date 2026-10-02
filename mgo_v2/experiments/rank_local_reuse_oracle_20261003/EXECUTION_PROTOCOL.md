# Frozen execution conventions

Owner plan `32a4146`. One CPU process, empty CUDA_VISIBLE_DEVICES, BLAS/OMP=1,
RLIMIT_AS=4 GiB before importing NumPy. No Torch/model/NCCL import and no GPU
process changes. Existing resident-model workers keep running. B8 is primary;
existing B16/B32 are secondary. Verify all raw receipt lengths and SHA256,
and trust rank0 global routes only with the committed cross-rank validation.

R0 uses the existing ReplicaReplay at rho0 and its independent slot-array
reference for all 432 events. Assert destinations, count matrices, first/reload
counts and complete cache state at every event. Decode totals must equal
committed F exactly. A compressed external archive saves each decode event's
routes, destinations, local/remote flags, exact send/receive matrices and full
before/after physical slot/LRU/primary state. No old result file is overwritten.

R1 keys are (layer,expert,requesting_rank). A remote occurrence is a key at one
decode event, with its route multiplicity. An individual candidate's marginal
saves one combine row per remote exact route and a dispatch row only when it
removes that token's last route to the old destination. Concentration and gate
use the sum of these **individual marginal** bytes as denominator. This sum
can differ from total actual traffic because dispatch savings interact when
multiple experts are moved. It is not a jointly achievable sum.

Future means strictly after the current step; current-event savings are excluded
from scores and persistence amortization. Same-layer steps 1..8 only; terminal
occurrences have no observed future. Record remote reuse and all-demand burst
statistics separately. Top shares use ceil(fraction * observed remote-pair count)
with descending weights. R2 recurrence share weights each current occurrence
by its marginal bytes if its next remote occurrence is within H. No extrapolation
past step8, and no H16. Gini is computed over observed remote pairs only.

R2 creates one independent candidate at the first remote occurrence of each
pair. Fetch once, persist with no capacity limit, and accumulate exact
single-candidate future marginals. Full per-occurrence and per-candidate records
are compressed externally; compact summaries and their hashes are committed.
The sum of individual candidate savings is not a joint multi-replica simulation.
Byte break-even compares peer bytes against one 9-MiB fetch for accounting only;
no equality of communication time or H2D time is asserted.

R3 is conditional on the B8 gate. Reuse the original placement/LRU implementation
with one opt-in candidate-score callback; its default greedy behavior and exact
current-traffic accounting remain unchanged. No separate cache implementation.
All selective cells begin with exactly the same rho0 prefill (no optional
prefill replicas). Replication starts in decode using future marginals frozen
from F. This focuses on decode reuse; old K/C retain their historical greedy
prefill and hence can have different decode-start cache states.

No extra rho quota is introduced; physical rank capacities remain 461/461/461/460.
Active keys are pinned; free slot first, else legal LRU with the existing key tie.
For victim lookahead, inspect the next H*48 layer events (or trace end), so an
upcoming layer in the current decode step is included. A unique victim with
future demand adds one predicted fetch; a surviving copy avoids that penalty.

Victim-penalty interpretation is recorded before any R3 cell: recommended
fetch-normalized score = future_peer_bytes / (1 + expected_victim_fetch).
This expresses bytes saved per projected fetch; it never subtracts H2D bytes
from peer bytes or assumes equal transfer costs. The owner was asked to resolve
this ambiguity; R0-R2 do not depend on the answer. If no alternative is selected,
use this stated interpretation. Rank candidates by descending score, then expert
ID and requesting rank. Accept strictly score > threshold (0/64KiB/256KiB/1MiB).
Recompute victim choices after every admission. All physical consequences affect
later events. Exactly four horizons times four thresholds, no tuning/repeats.

Replica reuse requires a later local demand served by that physical copy;
service at admission does not count. Track lifetime in layer events and divide
by48 for a fractional decode-step duration. Surviving replicas are right-censored
at trace end. Report censoring and avoid interpreting these as complete lifetimes.
Victim-induced reload counts mandatory fetches after a replica admission evicted
that key's last physical copy; predicted penalties and realized reloads are
reported separately. Net peer savings per replica fetch are relative to F and
include all trajectory effects.

Compare B8 against all committed nonzero-rho points for MODEST/NO_HEADROOM;
use the exact F/K thresholds from PLAN.md for STRONG_HEADROOM. Report B16/B32
coordinates without fabricated historical frontier points. CPU bytes only;
no physical follow-up is authorized.
