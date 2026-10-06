# Strict-Phase LA+CA Headroom Study

Status: IMPLEMENTATION / HEADROOM SEARCH. This packet is intentionally an oracle/stress characterization, not an unbiased average-case benchmark.

Date: 2026-10-06
Branch: `codex/policy-regime-20261005`

## 1. Question

How much can runtime miss placement beat Balanced Random (BR) when the workload, DP/rank assignment, and BR random seed are deliberately selected to expose BR's largest weakness?

The candidate is `LA_CA_NEAR`:
- load balance is primary;
- communication locality is used only among destinations whose projected critical-rank expert-row load is within 2% of the best load choice;
- balanced mandatory-miss H2D quotas are preserved;
- substitution, replication, and speculative prefetch are OFF.

The study answers a GO/STOP question about maximum placement headroom. It is not a paper-final neutral workload comparison.

## 2. Strict execution substrate

For both prefill and decode characterization use:

```
routing / admission
 -> enqueue all current required H2D
 -> finish all local required H2D
 -> GLOBAL H2D BARRIER
 -> forward A2A dispatch
 -> expert compute
 -> finish all local expert compute
 -> GLOBAL COMPUTE BARRIER
 -> return A2A
 -> combine
```

No H2D/expert overlap, no H2D/communication overlap, no speculative prefetch.

The forward dispatch necessarily precedes expert compute because destination ranks need remote activations. The compute barrier isolates return communication from rank completion skew.

This substrate is intentionally conservative and diagnostic. Optimized overlap can be restored only after the placement mechanism is established.

## 3. Candidate controller

For every current-layer miss expert, in descending total routed-row demand:

1. preserve BR/LA's balanced miss-count quota per rank;
2. estimate current expert-row load on every rank;
3. evaluate projected global maximum load if the expert is placed on each available rank;
4. find the minimum projected max load;
5. admit ranks within 2% of that minimum as near-tie candidates;
6. among near-tie ranks, choose maximum local demand for that expert (cheap CA/locality proxy);
7. tie-break by lower projected critical load, lower destination load, then rank id.

This is `O(num_misses * R^2)` with R<=8. It does not scan all tokens for every candidate rank and therefore avoids the high controller cost of dynamic packet-mask CA/FCA.

Primary objective: minimize max-rank expert service load.

Secondary objective, only inside the 2% load-safe region: maximize local expert demand, which reduces remote expert-route traffic.

## 4. Primary physical matrix

Cache: C30.
Cold expert cache at the start of measured prefill.
Qwen3-30B-A3B-Instruct-2507, BF16.
No substitution / replication / prefetch.

| world | local batch | context |
|---:|---:|---:|
| 4 | 16 | 256 |
| 4 | 16 | 512 |
| 4 | 64 | 256 |
| 4 | 64 | 512 |
| 8 | 16 | 256 |
| 8 | 16 | 512 |
| 8 | 64 | 256 |
| 8 | 64 | 512 |

Primary metric: TTFT.

Also record:
- all-rank H2D barrier duration;
- forward A2A duration;
- expert-compute local completion and compute-barrier duration;
- return A2A duration;
- per-rank expert rows;
- per-rank mandatory H2D copies/bytes;
- remote expert routes / token-rank packets where available;
- controller wall time.

## 5. Deliberately adversarial BR seed search

The search variables are intentionally independent.

- `sample_seed`: which R*B conversations are selected.
- `dp_seed`: how those selected conversations are assigned to ranks.
- `br_seed`: BR's random miss-expert-to-balanced-slot placement.

`LA_CA_NEAR` is deterministic for a fixed workload/rank assignment, so do not invent a candidate placement seed.

The objective is explicitly to maximize candidate gain over BR. We are allowed to choose a BR-unfavorable random seed because this stage asks for maximum realizable headroom.

Every physical comparison still uses identical requests, rank assignment, routing, cache capacity, current-layer miss set, total routed rows, and mandatory H2D quota. Only the miss-expert-to-rank assignment differs.

### S0-A: sample-potential screen

For each (R,B,L):
- sample_seed = 0..255;
- score global per-layer expert-demand skew / heavy-expert concentration;
- retain top 8 sample seeds.

### S0-B: DP + BR stress search

For every retained sample:
- dp_seed = 0..63;
- br_seed = 0..63.

For each exact route replay compute:
- BR sum of per-layer max-rank expert rows;
- LA_CA_NEAR sum of per-layer max-rank expert rows;
- BR and candidate remote expert-route traffic;
- balanced H2D quota validation.

Rank triples primarily by maximum relative reduction in max-rank expert rows:

`G_load = 1 - L_crit,candidate / L_crit,BR`

Tie-break by communication improvement, then deterministic seed order.

Keep top 8 distinct (sample_seed, dp_seed, br_seed) triples per physical cell.

This is an explicit BR-stress/oracle search.

## 6. Physical selection and validation

### S1
For each of the top 8 triples:
- paired BR and LA_CA_NEAR;
- one clean unprofiled sample each;
- counterbalance order;
- choose the triple with maximum measured TTFT gain: `G_TTFT = 1 - TTFT_candidate / TTFT_BR`.

### S2
For the physically best triple per cell:
- fresh process;
- two paired repeats;
- third repeat when the existing stability rule requires it;
- separate diagnostic pass for phase breakdown.

Final output reports both:
1. maximum measured TTFT gain over the deliberately stressed BR;
2. LA_CA_NEAR vs plain LA on the same selected triple as a secondary attribution only.

The GO/STOP decision is based on (1), because this packet's purpose is maximum headroom over the current strong baseline.

## 7. Context capture

The existing ShareGPT_LONG512 request manifest is reused.

For L=512:
- reuse exact frozen per-request route captures.

For L=256:
- use the last 256 valid tokens of the same frozen conversations;
- recapture routes with the exact model;
- do not derive L256 routes by slicing the L512 router trace, because hidden states/routing depend on the changed attention context.

## 8. R4 / R8 resource rule

R4 on the current shared machine may use physical GPUs 0,1,4,5 only.

Physical GPUs 2,3,6,7 on the current machine belong to another user and MUST NOT be touched.

Therefore:
- R4 physical runs may target 0,1,4,5.
- R8 CPU/trace seed search is valid immediately because it is GPU-independent after route capture.
- R8 physical timing must run only on a separately authorized eight-GPU allocation/server. The R8 runner must require an explicit 8-GPU physical list and must not silently default to 0..7 on the current machine.

## 9. Stop / interpretation

This is a maximum-headroom test.

- >=5% stable TTFT gain in at least one meaningful R/B/L regime: strong continuation signal.
- 2-5%: marginal; inspect scaling with R and phase breakdown.
- <2% even after the BR-adversarial search: strong evidence to stop or de-emphasize this placement axis.

Do not present the stress-selected number as dataset-average performance.
