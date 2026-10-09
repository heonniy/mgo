# Previous-winner batch and output-length expansion

Owner requested local batches32 and64 with a renewed best-input search, and
output lengths128 and256 in the existing configuration. Run local
B16/B32/B64 × output64/128/256; retain the historical completed B16/O64
five-repeat experiment and add the other eight cells. Local batch is per
rank/GPU; global requests are64/128/256. The historical harness calls output
length its decode length:64/128/256 generated tokens include the first token
from prefill, followed by63/127/255 decode forwards. Report both explicitly.

Preserve Qwen3-30B-A3B-Instruct-2507 BF16, ShareGPT first eligible512-token
inputs, C30/1843 physical slots, MAIN[459,459,459,458], P2/rank, greedy seed42,
no EOS stopping, no prefetch/substitution/replication. Both R-NEAR/G-NEAR use
grouped decode/native prefill and native C++ metadata/history/controller.
The mandatory order remains metadata/PLAN complete → forward A2A complete →
H2D complete → grouped expert compute complete → return A2A complete, with
all-rank Gloo completion gates. No H2D overlaps expert compute or NCCL.

Use only GPUs0,1,4,5; groups[0,1] and[4,5]. Two GPU ranks share each54GiB
fully pinned NUMA-local pool;108GiB unique source and128GiB host headroom.
Keep the existing lease/lock and burn transitions. Increase only the grouped
activation workspace bound from global64×topk8 to the requested global
batch×topk8:8.25/16.5/33MiB per rank; expert-cache capacity is unchanged.
Full requested-length warmups must establish finite outputs, grouped weighted
relative L2≤1%, exact metadata wire/layout checks, and no timing-time JIT.

Reuse the previously verified whole-corpus census:94145 source rows,
67333 eligible512-token records. Freeze deterministic random/coherent/mixed/
conversation-family candidates at each local batch before physical timing.
Each batch has32 constructed candidates plus the previous-winner control;
remove an exact duplicate ordered batch before nomination. Every batch has
distinct source rows and distinct512-token inputs, disjoint from its warmup.
Never pad by copying requests to enlarge the batch. Record original conversation
families separately; different split records need not be independent conversations.

The previous winner family_3 is a REQUIRED control. B16 retains its exact64
ordered inputs. B32/B64 retain those64 and add distinct requests, first from
the same original family then a deterministic whole-corpus fill. Larger batches
repartition inputs across ranks; they are not the original rank-routing trace.
Include this control in both screen and final even if it loses search selection.

Once per batch, collect a16-output G-routing nomination with lossless capture
and exact native C++ independent cold-cache R/G replay. Nominate top8 by
critical-group fetch-count proxy, plus the control if absent. This is a proxy,
not a serving result. For EVERY output length, separately screen all nominated
inputs with one fresh full-length unprofiled R/G pair, alternating arm order.
Freeze the top3 plus required control before final timing. Run all finalist
arms in counterordered three repetitions; if ANY (max−min)/median TPOT>5%,
add two repetitions to ALL finalist arms. Preserve every timing. Afterwards,
collect one separate all-rank phase/assignment diagnostic per finalist arm.

Validate full48×output event traces, exact same-arm token/cache/trace parity
across screen/final/repeats/diagnostics, per-event legal actual fetches and
quotas, cache/no-background-copy/source sharing/page locality, and physical
global serialization. Recompute endpoints and TPOT using output−1.
Report TTFT/TPOT/E2E and every repeat; primary median/mean/SD/range; expert
traffic, live cache trajectories, token agreement and separate TPOT breakdown.
Report fixed previous-winner controls and independently searched winners side
by side. State best OBSERVED among screened candidates, never corpus optimum.

Commit each completed candidate/arm experiment separately, then common
reports, and push at stage boundaries AFTER GPU workers exit. Run no CPU
builds/import-heavy analysis/Git packing during primary generation. Data/model/
raw input manifests remain under/data2/esjung. Preserve failures, diagnose and
repair without deleting failed attempts or selecting away slow repetitions.

Priority order: B32/O64, B64/O64, B16/O128, B16/O256, B32/O128, B32/O256,
B64/O128, B64/O256. Other original-plan baseline jobs remain paused.

```sh
/data2/esjung/envs/mgo-pcie/bin/python -u mgo_v2/scripts/run_pcie_maxgain_expansion.py \
  --prepared /data2/esjung/datasets/pcie_maxgain_expansion_20261010 \
  --out /data2/esjung/mgo-results/pcie_topology_ablation_20261009/maxgain_expansion_attempt1
```
