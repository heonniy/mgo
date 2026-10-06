# Owner-requested pure LA versus LA_CA_NEAR clean timing

One workload: R4 physical GPUs 0,1,4,5; C30; local B64; input512; decode32.
Run pure LA (controller policy4) and existing LA_CA_NEAR (policy7), one clean
MEASURE generation each after one warm correctness pass per policy. No phase
probes, profiling, BR run, new seed search, trace capture or matrix expansion.
Use H0/full-pinned V3 P2/T2, BF16, original frozen inputs and selected seed28.
Policy applies to both prefill and decode; cache carries over from prefill.
Report TTFT, TPOT and E2E separately with single-shot limitations.

Use existing shared worker and supervisor. The worker now labels its baseline
and candidate from its policy tuple, retaining the original BR/Near defaults.
Pure LA receives its own independent CPU replay proof. Existing Near proof,
input hashes, deterministic settings, fixed CPU affinities, no-compile gate,
state/role/copy-count checks and warm/measurement token parity stay enabled.
Keep cross-policy token differences; do not require cross-policy bit parity.

Start only with >=384GiB available host memory; stop below96GiB. Maximum30min.
Stop/restore only owned model-forward idle loads on0,1,4,5. Do not touch2,3,6,7.
Preserve all rank records, commit checkpoints and publish results on the current
codex/main-table-global-workload-20261006 branch.
