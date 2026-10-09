# PCIe / NUMA Group-Aware MoE Offloading: Experimental Plan

Status: PLAN ONLY (not implemented, not benchmarked, no GPU measurements).
Date: 2026-10-09
Repository: heonniy/mgo
Isolated branch: codex/pcie-topology-ablation-20261009
Base: codex/main-table-global-workload-20261006 @ 6433242dd7a741836f9338d7cd4433b1d61e9959

## 0. Goals and hypotheses

Evaluate whether assigning mandatory CPU-to-GPU expert fetches by physical PCIe/NUMA group, rather than distributing the remainder to the lowest-numbered ranks, decreases exposed fetch makespan and end-to-end decode latency. Separately test which *experts* should occupy those fixed rank quotas: random BR, existing communication-aware CA, NUMA/PCIe-weighted communication-aware CA, and existing load-constrained Near. Preserve existing runtime/model/eviction/cache parameters except for explicitly controlled ablations.

H1 (physical): With comparable effective aggregate H2D bandwidth for NUMA groups A and B, an M=6 miss pattern 4:2 groups (2,2,1,1 on physical GPUs 0,1,4,5) can be slower than a 3:3 pattern (1,2,1,2). This is a *hypothesis*, not measured evidence. Validate topology, CPU-pinned source NUMA placement, and bandwidth first.

H2 (Stage I): Group-balanced quotas lower critical-group H2D time versus rank-order residual assignment, for events with a relevant quota difference. Even if H2D improves, the impact on TPOT can be small; report that result.

H3 (Stage II): Given exactly the same group/rank quota, minimizing cross-NUMA EP traffic can reduce communication, but may increase expert compute skew. Near minimizes projected critical-rank compute under a bounded locality tie-break. Compare all alternatives without modifying their quotas.

An analytical caveat: for four homogeneous ranks split into two 2-GPU groups, the current rank-order quota and a count-balanced group quota differ at group level primarily when M mod 4 = 2. At M=6 they are 4:2 vs 3:3; at M=62 they are 32:30 vs 31:31. Any claimed absolute/relative improvement must be based on physical measurements, not the count ratios alone. If bandwidths differ, equal counts per group may not be optimal; explicitly evaluate bandwidth-weighted quotas as a follow-up only.

## 1. Machine and immutable scope

- Intended machine: the *new PCIe-only dual-socket host* described by the owner's lspci and nvidia-smi topo -m output. This is not assumed equivalent to the prior NVSwitch/P2P environment.
- Allowed physical GPUs: 0, 1, 4, 5 ONLY. Never initialize, allocate on, test NCCL with, stop processes on, or alter GPUs 2,3,6,7.
- NUMA group A = physical GPUs [0,1], NUMA node 0; group B = physical GPUs [4,5], NUMA node 1. Within each pair nvidia-smi reports PIX, between pairs SYS. The two sockets' PLX switch trees are separate, but shared bandwidth constraints must be established empirically.
- Logical ranks: R0->GPU0, R1->GPU1, R2->GPU4, R3->GPU5. The order matters to the rank-order remainder policy.
- Runtime preflight: verify PCI bus-ID mapping, CPU/socket affinity, link width/speed, IOMMU/NCCL transport, host-memory headroom, process/GPU isolation, BF16, model/checkpoint paths, and CPU pinned-buffer placement. Fail closed if any assumption differs. Never kill other users' processes.
- CPU affinity must be derived from this host's node CPU sets, not the old /home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json. Pin and first-touch host source pages on the intended NUMA node, inspect /proc/<pid>/numa_maps where feasible; also test actual runtime source placement.
- GPU/model work must run serially and return all resources after each job. Save nvidia-smi snapshots and checksums, seeds, git SHA, CUDA/PyTorch/NCCL builds, PCIe topology, CPU affinity and any transport override. No timing results are authorized or claimed by this PLAN commit.

## 2. Frozen model and workload: ONE headline cell

- Qwen3-30B-A3B-Instruct-2507; BF16; top-k routing unchanged; substitution OFF; replication OFF.
- GPU cache C30: 1,843 x 9 MiB global expert slots = 17,392,730,112 bytes. Respect the established MAIN+reserved physical slot accounting (e.g. MAIN 459,459,459,458 and reserved 2 per rank under the native headline stack); unused reserved slots still count against C30. Do NOT silently increase cache budget when turning prefetch OFF.
- Four GPUs; local batch 16 requests/rank = global batch 64; prompt/input length 512; exactly 64 generated tokens (63 decode intervals); greedy decoding, no premature stop on EOS.
- Reuse frozen ShareGPT-long target/warmup request manifests for R4_C30_B16_L512_O64 at mgo_v2/experiments/main_table_global_workload_20261006/expanded_matrix/manifests/. Relevant source WORKLOADS.json is in the same expanded_matrix directory. Rebase its existing absolute /home/hwlee/mgo-main-table/... manifest paths onto the actual checkout, **verify the recorded SHA256 hashes**, and use identical global request IDs/partitioning across all systems. Do not regenerate/change dataset when comparing policies. Existing target SHA256: dea64853847b1ba1d4f3e0c8cc112eb8ba84dcde5f1d04fb90adb6fca91d67d6; warmup SHA256: cf2aeb54933d729f55b8169ca9f0a5a5f951ce773d9d9d63490cfe4105e752bd.
- Cold dynamic expert cache/controller state at measured request-batch release; same prefill warms the cache naturally before decode; do NOT reset cache at the prefill/decode boundary. Keep compiled kernels/pinned CPU stores loaded between repeats if validated.
- Main executor: current audited native BF16 implementation, with the same eviction, memory footprint and model path for all OURS policies. Prefetch OFF for isolation. Keep any unchanged existing runtime optimization clearly logged.
- Three **unprofiled** repeats minimum for the primary one-cell comparisons after disjoint warmup; if timing stability is insufficient, predefine an additional two repeats for every comparable arm. Preserve ALL raw repeats; publish median, mean, sample SD, min/max and per-repeat values, not best-of-run selection.

## 3. Decode phase isolation: G2G communication BEFORE demand H2D

Critical requirement: the PCIe-domain fetch measurement must not inadvertently include simultaneous NCCL GPU-to-GPU traffic. For the *isolated* OURS mode, enforce the following causal sequence **on every decode MoE layer**:

1. Finish attention/router for the layer. Complete global routing/cache metadata all-gather and any related G2G metadata exchange on every rank; synchronize their streams.
2. All ranks derive the identical global miss set, count M, group quotas, and Stage-II expert destinations; verify deterministic PLAN checksum.
3. Perform forward expert dispatch all-to-all over G2G (based on known destinations), then **wait for completion and synchronize CUDA streams on all ranks**; perform the required global phase rendezvous before timed demand H2D. No NCCL collective may remain in flight.
4. Issue **mandatory H2D expert copies only**, using the pinned CPU source buffers; wait for all local copies, then record per-rank completion and globally align before expert compute. No prefetch, NCCL all-gather/dispatch, or G2G overlap with the measured H2D window. Distinguish local CUDA event copy service from all-rank last-ready wall time / barrier wait.
5. Expert compute; synchronize and optionally barrier to isolate return waiting.
6. Return/combine all-to-all. This return cannot physically precede H2D because it depends on expert output; therefore isolate it AFTER compute, not before fetch. Then advance to the next layer.

DO NOT reuse the present strict_serialized_phases sequence unchanged: in mgo_v2/mgo_v2/decode_runtime.py it currently runs H2D before dispatch and asserts arena_budget == 0. Implement a distinct guarded mode such as pcie_g2g_first_serial with dispatch-before-H2D and physical-slot-preserving reserved P2. Keep normal selected/main runtime mode unaffected. Add assertions that no pending priority-H2D prefetch or NCCL work overlaps the isolated H2D interval. Metadata collection timing, dispatch service, isolated H2D service, exposed fetch wait, expert compute, return service/wait, controller overhead and full layer latency must be separately instrumented; overlapping CUDA event spans must not be summed as independent time.

The isolated mode is a **mechanistic ablation**, not a drop-in measurement of the original overlapping production runtime. Run one extra current overlap/native Near-vs-group Near comparison if feasible to determine whether benefits survive overlap. Do not imply external baselines implement the forced phase order: their native serving pipelines are measured as-is and clearly labelled.

## 4. Experiment 1: Physical H2D / PCIe-group microbenchmark (priority 1)

Workers must use only GPUs0/1/4/5; one contiguous pinned BF16 expert payload = 9 MiB (9,437,184 bytes) per fetch. Use CPU memory local to each group's socket where possible; record page locality. Allocate and pre-touch *before* timing. Host source, H2D bytes, CUDA stream policy and synchronization boundaries should match the real OURS loader as closely as practical. Run warmups and at least 30 independent timed samples per cell; rotate test order (ABBA or randomized paired order), monitor interference and link state. No NCCL traffic during timed H2D; start after the common phase synchronization.

Test counts are in PHYSICAL GPU order [0,1,4,5]:

A. 2-fetch pair tests, identical total bytes:
- [2,0,0,0]: two copies through a single GPU in group A.
- [1,1,0,0]: one copy each on both GPUs in group A (shared PCIe-group path).
- [1,0,1,0]: one GPU in A and one GPU in B (split NUMA groups).
- [0,0,1,1]: both GPUs within B.
- [0,0,2,0]: two copies through a single GPU in B.
- Include symmetrical variants to eliminate individual-GPU asymmetry; record per-GPU and group time, NOT just summed bandwidth.

B. 6-fetch causal comparison, same six payloads in each arm:
- Rank-order: [2,2,1,1], groups 4:2.
- Group-balanced: [1,2,1,2], groups 3:3.
- Group-balanced alternative: [2,1,2,1], groups 3:3.
- Reversed skew: [1,1,2,2], groups 2:4.
- An additional [0,3,0,3] / [3,0,3,0] diagnostic is permitted only as a labelled per-GPU hotspot stress (NOT a per-rank-balanced policy).

C. Count-scaling: M=2,4,6,8,10,14,50,62 (and optionally trace-matched counts), comparing rank-order vs group-balanced under identical aggregate H2D bytes. M=4/8 are negative controls whose group totals already match. Sweep copy concurrency and note that host memory bandwidth, DMA engines, and per-GPU link may dominate before a shared PCIe uplink.

D. Realistic contention controls: local vs verified remote-NUMA pinned host buffer, isolated-H2D vs explicitly overlapped G2G diagnostic. The isolated measurements remain the main evidence; overlap is a secondary sensitivity analysis.

Report all median/p10/p90/p99 local H2D completion times, common-start-to-last-rank-ready makespan, group A/B achieved aggregate bandwidth, speedup and confidence/variation, plus topology and CPU memory placement receipts. Do not assume 3:3 is faster until observed. Stop before model runs if device protection, memory, or phase isolation fails.

## 5. Stage I quota definition and Experiment 2: Rank-order vs group-balanced (priority 2)

Same M global unique *mandatory* miss experts; no replication. Preserve routing, owner state and priority eviction semantics. Quotas are per event/layer and fixed before Stage-II placement.

Q_rank_order: q[r] = floor(M/4) + 1[r < M mod 4], with rank order [GPU0,GPU1,GPU4,GPU5]. At M=6 => [2,2,1,1] => groups 4:2.

Q_group_balanced: first assign M expert fetches across groups A={R0,R1} and B={R2,R3} as equally as integers permit (odd residual group priority rotates deterministically by layer/event; no future information). Next balance within each 2-GPU group; rotate per-rank remainder deterministically as well. For M=6 the pre-registered example is [1,2,1,2] => groups 3:3 (other 3:3 rotations accepted by prespecified tie-break). Enforce sum(q)=M, groups differ by <=1, ranks within a group differ by <=1, q>=0 and no slot/admission violations. Compare against rank-order with identical tie-break policy and log actual q vector every event.

Evaluate Q_rank_order vs Q_group_balanced with EXACT SAME Stage-II Near policy, cache size, dataset, seed and isolated G2G-first transport. This is the clean causal Stage-I test. Additionally report the naturally observed global-miss count histogram, % events with M mod 4 = 2, aggregate per-group H2D bytes, exposed fetch wait and impact on TPOT. The greedy per-event quota affects future cache states; record first-fetch/reload/eviction counts and total H2D bytes to separate direct placement from trajectory effects. Use optional frozen-routing replay to hold the input demand fixed without representing the replay as an independent model serving result.

If unequal effective group bandwidth is established, separately assess B_weighted (minimize max(Q_A/B_A,Q_B/B_B)) as a supplemental policy; do not silently substitute it for equal-count group balancing.

## 6. Stage II ablation under the same fixed group quota (priority 3)

Use identical Q_group_balanced, same miss set/trace, same cold-cache/reset, same H2D order, same hardware and exact C30 budget. A group-balanced policy MUST have the same per-rank quota at each event, regardless of the expert-to-rank assignment rule.

- G-BR (group-balanced random): random permutation of missing experts over the fixed quota slots, seeded/reproducible; no locality objective. This tests whether balancing fetch counts alone suffices.
- G-CA (original communication-aware): fill exactly the same quota slots using the existing communication/local-demand objective (use whichever CA/CA_NATIVE audited path is actually selected; record it), without explicit NUMA pricing.
- G-NUMA-CA (PCIe/NUMA-communication-aware): choose expert destinations inside the identical quota to penalize cross-group token-to-rank exchanges more heavily than same-group exchanges. Measure the actual communication path/cost (PIX within, SYS across) before freezing weights; report cross-NUMA remote route/packet volume and NCCL completion. Preserve the exact quota and mandatory H2D count. Explicitly distinguish *G2G communication cost* from H2D; peer path weights should come from measurements, not the GPU index alone. Compare G-CA to G-NUMA-CA to isolate topology-weighted communication placement.
- G-NEAR: existing load-constrained Near expert assignment (critical-rank projected expert work, then locality tie-break) with the same group-balanced quota. Do not retune Near's load slack on evaluation data.
- Existing R-NEAR, R-BR, R-CA: original rank-order quota baselines; needed to attribute Stage I and Stage II effects. R-NEAR vs G-NEAR holds Stage II fixed. Within G-* comparisons, hold Stage I fixed.

Do not conflate group fetch balance (Stage I) with locality-aware EP routing (Stage II). If G-NUMA-CA reduces cross-NUMA bytes but loses TPOT due to expert/return stragglers, report both effects. Log total and cross-group G2G bytes, remote token-rank pairs, expert groups launched and their rank skew, per-layer maximum expert-compute completion and return-A2A waiting/completion, controller time, H2D, hits/evictions, TPOT/E2E/TTFT, and token accuracy/parity or divergence where BF16 rounding differs.

Pre-register primary comparisons: R-NEAR vs G-NEAR; G-BR vs G-CA vs G-NUMA-CA vs G-NEAR; R-BR vs G-BR and R-CA vs G-CA for consistency. Controlled primary table should include these seven policy arms unless a documented preflight gate fails.

## 7. Experiment 3: Single-setting main comparison incl. external baselines (priority 4)

Single frozen cell: R4_C30_B16_L512_O64, Qwen3-30B-A3B-Instruct-2507 BF16, global 64 requests. No full model/dataset/cache sweep. Main evaluation one-cell table:
1. OURS G-NEAR (group-balanced quota; primary proposed arm).
2. OURS G-BR (group-balanced random).
3. OURS G-CA (same quota, existing CA).
4. OURS G-NUMA-CA (same quota, group-cost communication).
5. OURS R-NEAR (rank-order quota, original Near).
6. Optional extra rows R-BR, R-CA in the ablation table; same single setting.
7. MoE-Infinity (explicitly labelled repaired/adapted Qwen3 baseline).
8. DeepSpeed ZeRO-Inference (CPU parameter offload, *not* paper's FastGen).
9. llama.cpp synchronous native batch / owner-selected balanced3 layer-placement baseline.

Core OURS ablation and microbench have priority. Run external baselines only after correctness and resource preflight, as time permits; if not executed, mark NOT RUN (never carry old-server measurements over as new results). Baseline native communication/overlap schedules cannot be claimed identical to the serialized OURS characterization mode; label that methodological difference, and preferably include a production-native-overlap OURS headline row so external system comparisons are like-for-like as possible. Compare fixed BF16/input/output/global requests and actual GPU expert-residency bytes; record host-pinned memory, total HBM and distinctions between dynamic expert caching, generic CPU parameter offload, and static layer placement. For llama.cpp use owner-selected balanced3, 32 threads, CUDA graphs OFF, graph reuse OFF, record unutilized static C30 budget; do not revert to legacy layer-tail placement silently.

Every primary requires exact 64 generated tokens, recorded TTFT, TPOT=(E2E-TTFT)/63 and E2E, peak GPU memory per device, host RSS and pinned bytes. Same frozen target requests and disjoint warmup for every system. If a framework's generated output differs numerically, retain result and explicitly document comparisons and sampling semantics.

## 8. Proposed implementation boundaries (no implementation included in this commit)

- mgo_v2/scripts/la_placement.py: allow externally supplied per-rank quotas for Near; preserve old API/default behavior for current experiments.
- mgo_v2/scripts/br_carep_cpu.py and existing CA/CA_NATIVE adapter: accept explicit quota slots; BR randomizes experts, not quota. Ensure Hungarian/greedy solvers respect fixed quota and do not implicitly rebuild rank-order slots.
- mgo_v2/scripts/env_offload_policy.py and mgo_v2/mgo_v2/controller.py: compute quota before candidate assignment and feed it to BR/CA/Near under explicit experimental mode; preserve original policy enum/ABI by default and verify prefetch OFF.
- mgo_v2/mgo_v2/decode_runtime.py: add separate G2G-first fully serialized decode path (all-gather metadata -> PLAN -> completed dispatch -> mandatory H2D -> expert compute -> return), with explicit reserved P2 accounting, no H2D/G2G overlap; expose per-phase receipts. Existing strict_serialized_phases path is **not** equivalent.
- mgo_v2/examples/headline_ours_worker.py: opt-in pcie quota policy and G2G-first flag; replace host-specific fixed_affinity file dependence with validated actual-host affinity; keep rank mapping exact.
- mgo_v2/scripts/run_headline_job.py: host-specific root/env/checkpoint paths must be parameterized, default to safe/serial execution, require explicit physical-GPU allowlist (0,1,4,5), validate target occupancy and reserve memory; never stop unrelated/GPU2,3,6,7 processes.
- New microbench worker/launcher, policy ablation launcher, frozen workload path relinker, test_quota.py and standardized results summarizer to be implemented on this branch in subsequent code task.
- External baseline workers are adapted only as required for valid paths/CPU affinity and 0,1,4,5 GPU visibility; preserve separate original installs and executable versions.

## 9. Acceptance gates, ordering, and reporting

G0: inspect GPU0/1/4/5 ownership, topology and NUMA locality, NCCL transport, environment and source manifests; reproduce hashes. Abort instead of touching other GPUs.

G1: CPU unit tests for M=0..256: exact miss-count conservation; deterministic rank-order / group quotas; M=6 rank-order [2,2,1,1], group [1,2,1,2] at registered rotation; odd-M rounding; legal cache slots; all policies honor identical supplied quota; group/rank owner state consistent after eviction; no replication; no substitution.

G2: microbench 2-fetch and 6-fetch matrices with synchronized H2D-only window, topology/NUMA verified, no collectives in flight, all timing samples saved. Publish a negative or positive result without changing the experiment matrix.

G3: one-token/full-64-token smoke tests in isolated runtime: matching global routing/PLAN checksum across ranks; finished dispatch before first demand H2D; all mandatory copies complete before expert compute; return occurs after compute; 2 G2G expert collectives per decode layer; no extra unplanned communication; no Numba/CUDA recompilation in timed region; consistent slot/owner states and finite logits; 0/1/4/5 exclusivity verified.

G4: run Stage-I Near pair first, Stage-II group-policy ablation second, only then external baselines. Use ABBA/interleaved repeats to limit drift, capture every attempt; time bound set before execution, use STOP file, no silent fallback or shrinking the batch/cache budget.

Deliverable artifacts for a subsequent execution:
- topology.json, host_numa_map.json, software_versions.json, source/checkpoint/workload hashes;
- microbench_raw.csv, microbench_summary.json, microbench_plot.pdf;
- quota_event_trace.csv (M, M mod 4, rank quotas, group totals, assigned experts, H2D bytes);
- phase_timing.jsonl, per-rank stage times, e2e policy raw repeats and main_table.csv/.tex;
- RESULTS.md explicitly distinguishing measured, modeled and unrun;
- commit SHA of each implementation/result revision and the exact executed command/environment.

## 10. Stop conditions and nonclaims

- Stop if tests cannot isolate NCCL/G2G from H2D, the new server's NUMA mapping differs, any forbidden GPU is accessed, any process/host memory guard fails, or the batch/cache budget becomes incomparable.
- No claim of 25% real speedup at M=6 without measurement. 4:2 vs 3:3 only establishes a plausible opportunity under an effective aggregate-bandwidth bottleneck.
- Do not claim the forced serialization is the production default, or that return A2A can run before expert compute.
- Do not call a generic DeepSpeed ZeRO CPU parameter offload configuration 'FastGen'.
- This PLAN commit writes no executable policy changes and reports no real GPU experiment; implementation, tests and execution require separate commits.
