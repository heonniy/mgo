# Controller / Cache / Fetch 흐름 검증 문서

> **목적**: 현재 구현된 `GlobalCacheController` + `GlobalSlotCacheView` + `CommandDispatcher` 가 의도대로 동작하는지 — 특히 **multi-process(EP=DP=world_size) 환경에서 모든 rank의 cache view가 byte-identical로 유지되는지**, fetch가 정확히 한 번씩 발행되는지, archer 실제 상태와 어긋나지 않는지 — 한 페이지에서 확인할 수 있도록 정리.
>
> **범위**: forward 한 step 안에서 1개 MoE layer가 처리되는 전 과정.

---

## 0. 멘탈 모델 한 줄

> **"모든 rank가 같은 컨트롤러 코드를 같은 입력으로 실행해서, 자기 rank 분량의 archer 명령만 발행한다. NCCL은 입력 동기화에만 쓰고, 결정 자체는 lock 없이 결정론적으로 모두가 같은 결론에 도달한다."**

이게 작동하려면 4가지 결정성 (determinism) 기둥이 필요하고, 코드는 그걸 강제하도록 짜여 있음. 아래에서 하나씩 검증.

---

## 1. 핵심 데이터 구조

### 1.1 `RankSlotCache` — 한 rank의 slot 배열

[controller/slot_cache.py:54-218](../moe_infinity_ep/controller/slot_cache.py#L54-L218)

```
RankSlotCache  (rank=r, cap=C)
├── slots:           [Optional[ExpertKey]] × C   # 물리 slot 1:1
├── expert_to_slot:  Dict[ExpertKey, int]        # 역인덱스
├── meta:            [Optional[SlotMeta]] × C    # last_used / freq / inserted_at
├── layer_seq:       int                          # bump_layer로 +1
└── touch_counter:   int                          # apply/touch마다 +1
```

**Slot = 물리 자원 1단위**. 같은 expert가 두 slot에 동시 존재할 수 없음 (Invariant I3).

#### `apply(slot, evict, insert, demand_count)` — 유일한 mutate 통로
[slot_cache.py:106-162](../moe_infinity_ep/controller/slot_cache.py#L106-L162)

3가지 합법 케이스:
| evict | insert | 의미 |
|-------|--------|------|
| `A`    | `B`     | full replace (slot[i]=A → slot[i]=B) |
| `None` | `B`     | fill (slot[i] 비어있던 거 채움) |
| `A`    | `None`  | pure evict (slot 비움) |

각 케이스에서 **3가지 invariant를 pre-check로 assert**:
- `slots[slot] == evict` ([L134-L136](../moe_infinity_ep/controller/slot_cache.py#L134-L136))
- `expert_to_slot[evict] == slot` ([L137-L140](../moe_infinity_ep/controller/slot_cache.py#L137-L140))
- `insert not in expert_to_slot` ([L149-L152](../moe_infinity_ep/controller/slot_cache.py#L149-L152))

→ 한 줄이라도 깨지면 즉시 `AssertionError`. **shadow가 깨지는 순간 바로 죽음**.

### 1.2 `GlobalSlotCacheView` — 모든 rank의 snapshot

[slot_cache.py:221-289](../moe_infinity_ep/controller/slot_cache.py#L221-L289)

```
GlobalSlotCacheView
└── per_rank: [RankSlotCache] × ep_size
```

각 rank의 controller는 **자기 rank 것만 보는 게 아니라 ep_size개 전체 RankSlotCache를 들고 있음**. 이게 multi-process 일관성의 핵심: 모든 rank가 "다른 rank들이 지금 어떤 expert를 어느 slot에 갖고 있는지" 알기에, cross-rank 통신 없이 동일한 routing/owner 결정 가능.

**핵심 query**:
- `locate(key)` — 그 expert를 가진 lowest rank, 없으면 None ([L242-L247](../moe_infinity_ep/controller/slot_cache.py#L242-L247))
- `total_slots_used()` — 통계용
- `bump_all_layers()` — 매 layer 끝에 모든 rank의 `layer_seq` +1 ([L263-L265](../moe_infinity_ep/controller/slot_cache.py#L263-L265))

### 1.3 `FetchOp` / `HitOp` / `LayerPlan` — 결정의 단위

[controller/layer_plan.py:19-95](../moe_infinity_ep/controller/layer_plan.py#L19-L95)

```python
HitOp     = (expert, owner_rank, slot)              # cache에 이미 있음
FetchOp   = (expert, fetcher_rank, dst_slot, victim_expert)  # 새로 가져옴
LayerPlan = layer_id + fetch_ops[] + routing_map_miss + expert_to_rank
```

`FetchOp`는 **fully-resolved** — dst_slot, victim이 박혀 있음. archer는 이 단위로 한 번에 atomic하게 처리.

---

## 2. Multi-process 일관성: 4가지 결정성 기둥

> 모든 rank의 `GlobalSlotCacheView` 가 **매 layer 끝에 byte-identical** 이어야 그 다음 layer 의 결정도 동일해진다. 이걸 lock 없이 보장하는 게 핵심.

### 기둥 1 — **NCCL all_gather로 입력을 byte-identical하게 동기화**

`DemandCollector.collect_with_count()` [demand_collector.py:49-79](../moe_infinity_ep/controller/demand_collector.py#L49-L79)

```python
local_count = router_mask.sum(dim=0).to(int64)        # [num_experts]
per_rank = empty([ep_size, num_experts], int64)
dist.all_gather(list(per_rank.unbind(0)), local_count, group=ep_group)
return per_rank.cpu()
```

- **모든 rank가 같은 `[ep_size, num_experts]` int64 행렬을 받음**
- `.cpu()` 로 옮겨 deterministic 정수 연산만 함 (GPU non-determinism 차단)

→ Phase 1 입력이 모든 rank에서 비트 단위로 같음.

### 기둥 2 — **OwnerPolicy.assign_rank() 가 결정론적**

[owner_policy.py:46-140](../moe_infinity_ep/controller/owner_policy.py#L46-L140)

세 정책 모두:
- 입력 `miss_keys`는 `sorted(miss_set)` 으로 정렬되어 들어옴 ([global_controller.py:155](../moe_infinity_ep/controller/global_controller.py#L155))
- 정책 내부 tie-break도 `(rank, idx)` 같은 결정적 key 사용
  - `BalancedPlacement`: `key=lambda r: (loads[r], r)` ([L102](../moe_infinity_ep/controller/owner_policy.py#L102))
  - `DemandAwareOwner`: `(-demand, layer, expert)` ([L132-L135](../moe_infinity_ep/controller/owner_policy.py#L132-L135))

→ 같은 (miss_keys, demand, cluster_state) → **모든 rank에서 같은 출력**.

### 기둥 3 — **RankPlanner.plan() 이 sequential & 결정론적**

[rank_planner.py:42-103](../moe_infinity_ep/controller/rank_planner.py#L42-L103)

```python
ordered = sorted(miss_experts, key=lambda k: (-demand[k], k[0], k[1]))
for expert in ordered:
    if rank_cache.is_resident(expert): continue
    empty = rank_cache.first_empty_slot()
    if empty is not None:
        dst_slot, victim = empty, None
    else:
        victim_slot = evict_policy.pick_victim(rank_cache, expert, demand)
        dst_slot, victim = victim_slot, rank_cache.slots[victim_slot]

    fetch_ops.append(FetchOp(expert, rank, dst_slot, victim))
    rank_cache.apply(slot=dst_slot, evict=victim, insert=expert,    # ★ 즉시 mutate
                     demand_count=demand[expert])
```

**`apply`를 매 iter 즉시 호출**하는 이유:
- 다음 iter의 `first_empty_slot()` / `pick_victim()` 이 **방금 install된 결과를 본 상태**에서 결정.
- 같은 빈 slot을 두 번 노리는 일 X.
- 방금 install한 expert를 곧바로 victim 후보로 보지 않음 (touch_counter가 +1 되어 last_used 가장 큼 = LRU 정책에서 victim 안 됨).

**`pick_victim`** [evict_policy.py:45-96](../moe_infinity_ep/controller/evict_policy.py#L45-L96): LRU/LFU/DA 모두 `score = (..., slot_index)` 로 tie-break → 결정론적.

### 기둥 4 — **모든 rank가 모든 rank의 plan을 produce**

[global_controller.py:198-209](../moe_infinity_ep/controller/global_controller.py#L198-L209)

```python
all_fetch_ops = []
for r in range(self.ep_size):        # ← 자기 rank만 X, 전체 rank loop
    misses_r = phase1.miss_per_rank.get(r, [])
    ops = self.rank_planner.plan(rank=r, miss_experts=misses_r,
                                 rank_cache=self.cache_view.per_rank[r],
                                 demand=phase1.demand)
    all_fetch_ops.extend(ops)
```

→ 모든 rank가 **자기 rank 뿐 아니라 다른 rank들의 RankSlotCache도 똑같이 mutate**.
→ Phase 2 끝나면 모든 rank의 `GlobalSlotCacheView` 가 byte-identical.

실제로 archer 명령은 자기 rank 것만 발행 ([command_dispatcher.py:123](../moe_infinity_ep/controller/command_dispatcher.py#L123) `op.fetcher_rank == my_rank` filter).

---

## 3. 한 Layer Forward 전체 Timeline

[runtime/ep_executor.py:149-377](../moe_infinity_ep/runtime/ep_executor.py#L149-L377) `_coop_dispatch()` 가 master orchestrator.

```
                    [Rank 0]               [Rank 1]      ...     [Rank N-1]
                       │                      │                       │
Phase 0  begin_layer  │  log only            │  log only            │
                       │                      │                       │
Phase 1  router→mask  │  router→mask         │  router→mask         │
                       │                      │                       │
         collect_with_count  ──── all_gather (NCCL) ───── 모든 rank가 같은 per_rank 받음
                       │                      │                       │
         hit/miss 분류 │  hit/miss 분류       │  hit/miss 분류       │  (모두 byte-identical)
         hit touch    │  hit touch           │  hit touch           │  → cache_view 동시 mutate
         owner.assign │  owner.assign        │  owner.assign        │  (deterministic)
                       │                      │                       │
                  ─── Phase1Output : 모든 rank 동일 ───
                       │                      │                       │
Phase 3a  hit a2a launch (NCCL)  ─────────────────────────────────
        pack_tokens → route_tokens                                    │ (background work 시작)
                       │                      │                       │
Phase 2  plan_misses (all ranks loop)
         shadow 즉시 mutate (rank_planner.apply)                      │
         → LayerPlan : 모든 rank 동일                                  │
                       │                      │                       │
Phase 3d  CommandDispatcher.launch_fetches(my_rank)                   │
          ▷ 자기 rank fetch_ops 만 archer 에 issue                     │
          ▷ explicit_replace_async (h2d_stream PCIe start, non-block) │
                       │                      │                       │
Phase 3e  miss a2a launch (NCCL)  ─────────────────────────────────  │  (fetch와 병렬)
                       │                      │                       │
Phase 3b  hit GEMM (archer compute_stream)                            │
          (cache에 있는 expert만 사용, fetch 대기 X)                    │
                       │                      │                       │
Phase 3g  miss GEMM (archer compute_stream)                           │
          fetch_event auto-wait (S6 의 ExplicitReplaceAsync)           │
                       │                      │                       │
Phase 3c/3h  combine a2a (NCCL 역방향)  ──────────────────────────────
             scatter_combine_into(final, send_back, token_idx)        │
                       │                      │                       │
Phase 4  verify_layer_end                                             │
         shadow == archer_physical assert                             │
         bump_all_layers (layer_seq += 1)                             │
                       │                      │                       │
                     return final            return final            return final
```

### 3.1 각 Phase의 mutate 책임자

| Phase | mutate 주체 | mutate 대상 | NCCL? |
|-------|-------------|-------------|-------|
| 1.1 demand 수집 | `DemandCollector` | (read only) | ✅ all_gather |
| 1.2 hit/miss 분류 | `Controller` | (read only — cache_view 조회) | — |
| 1.3 hit touch | `Controller` | `RankSlotCache.touch_use()` (meta only) | — |
| 1.4 owner 분배 | `OwnerPolicy` | (read only) | — |
| 2 plan_misses | `RankPlanner` | **`RankSlotCache.apply()` — slots/expert_to_slot/meta** | — |
| 3a/3e/3c/3h a2a | `nvlink_router` | (read only) | ✅ all-to-all |
| 3d fetch | `CommandDispatcher` | archer 내부 (`cached_experts_`, `cache_sizes_`) | — |
| 3b/3g GEMM | `archer` | (computation only) | — |
| 4 verify | `Controller` | `bump_all_layers` (`layer_seq` +=1) | — |

핵심: **`cache_view` mutate 는 Phase 1.3 (meta only) + Phase 2 (slot replace) + Phase 4 (layer_seq) 세 곳뿐**. 다른 곳에서 `cache_view.per_rank[r].apply()` 호출하는 코드가 있다면 invariant 위반.

---

## 4. Cache State와 archer Physical 의 동기화

> **shadow** = `cache_view` (controller 가 만든 의도) 
> **physical** = archer 의 `cached_experts_` (실제 GPU 메모리 상태)

### 4.1 shadow → physical 흐름 (Phase 2 → 3d)

```
Phase 2: rank_planner.apply(slot=5, evict=(L,A), insert=(L,B))
   └─ cache_view.per_rank[me].slots[5] = (L,B)   (★ 즉시)
   └─ FetchOp(expert=(L,B), dst_slot=5, victim=(L,A)) 생성

Phase 3d: CommandDispatcher.launch_fetches(plan, my_rank=me)
   └─ self._has_replace_async 면:
      └─ local_dispatcher.explicit_replace_async(
            0, v_layer=L, v_expert=A, n_layer=L, n_expert=B
         )
         → archer 측에서 cached_experts_ 에 (L,B) 추가 + (L,A) 제거 atomic
         → h2d_stream 에 cudaMemcpyAsync(dst=slot5 의 GPU buffer, src=host pinned)
         → fetch_event record (Phase 3g GEMM 이 wait)
```

### 4.2 두 경로 (S6 atomic vs legacy pair)

[command_dispatcher.py:127-132](../moe_infinity_ep/controller/command_dispatcher.py#L127-L132) 가 capability detection:

| 우선 | 조건 | 경로 |
|------|------|------|
| 1순위 | `_has_replace_async` ON & `_force_sync` OFF | `explicit_replace_async` — atomic |
| 2순위 | `_has_fetch_async` ON & `_force_sync` OFF | `explicit_evict` + `explicit_fetch_async` 페어 |
| 3순위 | `_has_fetch_sync` ON | `explicit_evict` + `explicit_fetch` (sync) — fallback |

→ 어느 경로든 **archer 측 `cached_experts_` 는 controller 가 의도한 (slot 단위) 변화와 정확히 일치**해야 함. 일치 안 하면 Phase 4 에서 잡힘.

### 4.3 Failure modes 가 정확히 카운트됨

[command_dispatcher.py:51-60](../moe_infinity_ep/controller/command_dispatcher.py#L51-L60):
- `n_fetches`, `n_evicts` — 성공 count
- `n_fetch_failures`, `n_evict_failures` — exception/false return
- `n_evict_noops` — idempotent skip (이미 없는 expert)
- `failure_examples[]` — 처음 N개 dump (kind, layer, expert, error)

→ counters로 노출되어 [ep_executor.py:319-325](../moe_infinity_ep/runtime/ep_executor.py#L319-L325) 에서 매 layer 통계 갱신.

---

## 5. Phase 4 verify — 잡아내는 시나리오들

[global_controller.py:240-272](../moe_infinity_ep/controller/global_controller.py#L240-L272)

```python
physical = set(local_dispatcher.get_cached_experts(0))   # archer 실제
drift_count, missing, extra = (
    self.cache_view.per_rank[self.ep_rank].verify_against_archer(physical)
)
# missing = shadow - archer  (controller 는 있다고 생각하는데 archer 엔 없음)
# extra   = archer - shadow  (archer 엔 있는데 controller 는 모름)

if drift_count > 0:
    raise RuntimeError(...)   # 새 design 하에선 정상 동작 시 불가능
```

### 어떤 버그가 잡히는가

| 시나리오 | 어디서 누락 | drift 증상 |
|----------|-------------|-----------|
| `apply` 호출 빠뜨림 (shadow는 안 바꾸고 archer 만 변경) | `RankPlanner.plan` | `extra ≠ ∅` |
| archer 명령 보냈는데 failure → physical 안 바뀜 | `CommandDispatcher` | `missing ≠ ∅` |
| `_force_sync` ON 인데 evict 가 실패 (continue) → physical 에 victim 남음 | [command_dispatcher.py:259](../moe_infinity_ep/controller/command_dispatcher.py#L259) | `extra ≠ ∅` |
| archer 가 자체 LFU eviction 으로 expert 자체 제거 | upstream archer | `missing ≠ ∅` (`MOE_EP_DISABLE_ARCHER_EVICT=1` 로 막아야 함) |
| 다른 rank 가 같은 expert 를 fetch (owner_policy bug) | `OwnerPolicy.assign_rank` | rank-level drift는 안 잡히지만 결과적으로 redundant fetch |
| 어느 rank 의 controller 가 다른 입력 받음 (all_gather race) | `DemandCollector` | 다음 layer 부터 모든 rank drift |

`MOE_EP_VERIFY_LAYER_END=1` 가 default ON ([global_controller.py:36](../moe_infinity_ep/controller/global_controller.py#L36)) → **실험 중에는 끄지 말 것**.

---

## 6. 의도대로 됐는지 자가 점검 체크리스트

이 항목들이 만족되면 구현이 의도와 일치한다고 볼 수 있음.

### 6.1 정적 (코드 grep) 점검

- [ ] `RankSlotCache.apply()` 호출이 **단 한 군데** 에서만 일어남
  ```bash
  grep -rn "\.apply(" moe_infinity_ep/controller/ | grep -v test
  ```
  → `rank_planner.py` 의 한 줄 ([L96-L101](../moe_infinity_ep/controller/rank_planner.py#L96-L101)) 만 나와야 함.

- [ ] `touch_use()` 호출이 Phase 1 의 hit touch 한 곳에서만
  ```bash
  grep -rn "touch_use" moe_infinity_ep/
  ```
  → `global_controller.py:148-150` 만.

- [ ] `bump_all_layers()` 호출이 Phase 4 한 곳에서만
  ```bash
  grep -rn "bump_all_layers\|bump_layer" moe_infinity_ep/ | grep -v test
  ```

- [ ] 모든 정책의 tie-break 가 slot 또는 (layer, expert) 포함 (random/dict iteration 의존 X)

- [ ] `_force_sync` 가 default OFF, `MOE_EP_DISABLE_ARCHER_EVICT` 가 default ON

### 6.2 런타임 점검 — heavy debug 켜고 1 step

```bash
MOE_EP_HEAVY_DEBUG=1 MOE_EP_VERIFY_LAYER_END=1 \
  torchrun --nproc_per_node=N scripts/m1_smoke.py configs/...yaml
```

각 layer 마다:
- [ ] `[ctrl rank=R] L_ P1: demanded=D hits=H miss_dedup=M miss_per_rank={...}` — **모든 rank 에서 demanded/hits/miss_dedup 값이 동일**해야 함 (NCCL all_gather 정상 작동)
- [ ] `[ctrl rank=R] L_ P2: total_fetch=F my_fetch=My shadow_after=S` — **total_fetch, shadow_after 가 모든 rank 동일**, my_fetch 는 rank 마다 다름 (자기 분량)
- [ ] `[dispatch rank=R] L_ fetches=My fetch_fail=0 evict_fail=0` — failure 0
- [ ] Phase 4 에서 `DRIFT` 로그 0회

### 6.3 카운터 점검 — N step 후

```python
counters.cache_hits / (cache_hits + cache_misses)   # hit rate
counters.dispatch_fetch_failures                    # == 0 이어야
counters.dispatch_evict_failures                    # == 0 이어야
counters.drift_layer_pre                            # 전부 0
counters.pcie_fetch_bytes / (n_layers * n_steps)    # rank 마다 ~동일 (LB 확인)
```

`drift_layer_pre` 가 단 한 layer 라도 0 아니면 — 그 layer 직전 어딘가에서 invariant 깨졌다는 뜻. heavy debug 켜고 재현.

### 6.4 결정성 외부 검증 — 두 rank snapshot 비교

```python
# 어느 layer 끝나고
snap0 = controller.cache_view.snapshot()  # rank 0
snap1 = controller.cache_view.snapshot()  # rank 1 (다른 process)
assert snap0 == snap1                     # ★
```

byte-identical 이면 4가지 결정성 기둥이 모두 성공한 것. 다르면 어느 기둥이 무너졌는지 (per_rank.snapshot() 비교로) 추적.

---

## 7. 잘못되기 쉬운 곳 (트러블슈팅)

### 7.1 "내 rank fetch 만 발행"의 missing 케이스

증상: drift 의 missing 에 자기 fetcher_rank 의 expert 가 등장.

원인 후보:
- `CommandDispatcher.launch_fetches` 가 `local_dispatcher=None` 으로 early return ([L121-L122](../moe_infinity_ep/controller/command_dispatcher.py#L121-L122)) — wiring 누락
- archer build 가 `explicit_*` API 가 모두 없음 → init log 의 `effective=none` 확인
- exception → `failure_examples` 에 첫 N개 기록

### 7.2 "다른 rank의 cache state 가 어긋남"

증상: 모든 rank 에서 P1 의 `hits` 가 다름.

원인 후보:
- `DemandCollector` 입력이 device-dependent (e.g. CUDA RNG, dropout) → router_mask 가 rank 마다 다름
- topk_softmax 가 비결정적 (사용 lib 확인)
- ep_group 이 잘못 구성 → 다른 group 의 rank 와 all_gather

### 7.3 "shadow 는 맞는데 hit rate 가 너무 낮음"

증상: drift 0 이지만 매 layer cache_miss 많음.

원인 후보 (구현 버그 X, 정책 미스매치):
- `owner_policy=balanced` 라 같은 expert 가 layer 마다 다른 rank → locality 손실. `static` 으로 재실험.
- cap_per_rank 가 너무 작음 — `sparse_hbm_ratio` 또는 explicit `cache_capacity_per_rank` 상향
- 워크로드의 demand 분산이 큰데 `evict_policy=lru` — `demand_aware` 시도

### 7.4 "Phase 2 가 빈 plan 만 만듦"

증상: `total_fetch=0` 인데 miss 있는데도.

원인 후보:
- `OwnerPolicy` 가 일부 rank 에만 분배 → 다른 rank 는 빈 list → ok 한 동작
- 모든 miss 가 이미 resident (Phase 1 의 miss/hit 분류 vs Phase 2 의 `is_resident(expert): continue` 사이의 race) — single-controller path 라면 발생 X. 두 controller 인스턴스가 있다면 위험.

---

## 8. 한 그림으로 정리

```
┌──────────────────────────── Rank r 의 한 process ────────────────────────────┐
│                                                                              │
│   model.forward(layer L)                                                     │
│        │                                                                     │
│        ▼                                                                     │
│   EPExpertExecutor.run_layer(L, hidden, gate, ...)                           │
│        │                                                                     │
│        ▼                                                                     │
│   _coop_dispatch(L, hidden, router_mask, ...)                                │
│        │                                                                     │
│        │  ┌── Phase 0 ────────────────────────────────────────────┐          │
│        │  │ controller.begin_layer(L)                              │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        │  ┌── Phase 1 ─────────────────────── NCCL all_gather ────┐          │
│        │  │ p1 = controller.classify_and_kick_hit(L, router_mask) │          │
│        │  │   ├ demand_collector.collect_with_count() ──★ NCCL    │          │
│        │  │   ├ for e in demanded: locate → hit/miss              │          │
│        │  │   ├ hit_ops 마다 cache_view.touch_use()  (meta only)   │          │
│        │  │   └ owner_policy.assign_rank(miss, demand, view)      │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        │  ┌── Phase 3a (hit a2a) ──── NCCL ───── (background) ────┐          │
│        │  │ hit_table = build_expert_rank_table(hit_routing)      │          │
│        │  │ hit_plan  = pack_tokens(hidden, mask, weights, table) │          │
│        │  │ hit_recv  = route_tokens(plan, ep_group) ★ NCCL       │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        │  ┌── Phase 2 ────────────────────────────────────────────┐          │
│        │  │ plan = controller.plan_misses(p1, L)                  │          │
│        │  │   for r in range(ep_size):    ← ALL ranks loop        │          │
│        │  │     ops = rank_planner.plan(r, miss[r], cache_view[r])│          │
│        │  │       for expert in sorted_by_demand:                 │          │
│        │  │         empty? → dst_slot=empty, victim=None          │          │
│        │  │         else  → victim_slot=evict_policy.pick_victim()│          │
│        │  │         FetchOp 생성                                  │          │
│        │  │         cache_view[r].apply(...)  ★ 즉시 mutate       │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        │  ┌── Phase 3d ──────────────── archer h2d_stream ────────┐          │
│        │  │ command_dispatcher.launch_fetches(plan, my_rank=r)    │          │
│        │  │   for op in plan if op.fetcher_rank == r:             │          │
│        │  │     explicit_replace_async(victim, new, slot)         │          │
│        │  │       → cudaMemcpyAsync host pinned → GPU slot        │          │
│        │  │       → fetch_event record                            │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        │  ┌── Phase 3e (miss a2a) ── NCCL ── (fetch 와 병렬) ─────┐          │
│        │  │ miss_plan = pack_tokens(...)                          │          │
│        │  │ miss_recv = route_tokens(...) ★ NCCL                  │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        │  ┌── Phase 3b/3g ────── archer compute_stream ───────────┐          │
│        │  │ hit_local_output  = _archer_batch_compute(hit_recv)   │          │
│        │  │ miss_local_output = _archer_batch_compute(miss_recv)  │          │
│        │  │   (miss 는 fetch_event auto-wait)                     │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        │  ┌── Phase 3c/3h ── NCCL 역방향 + scatter ───────────────┐          │
│        │  │ hit_send_back  = return_outputs(...)                  │          │
│        │  │ miss_send_back = return_outputs(...)                  │          │
│        │  │ scatter_combine_into(final, ..., token_idx)           │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        │  ┌── Phase 4 ────────────────────────────────────────────┐          │
│        │  │ drift, missing, extra =                                │          │
│        │  │    controller.verify_layer_end(L)                      │          │
│        │  │   ├ physical = local_dispatcher.get_cached_experts(0)  │          │
│        │  │   ├ assert shadow == physical (drift=0)                │          │
│        │  │   └ cache_view.bump_all_layers()                       │          │
│        │  └────────────────────────────────────────────────────────┘          │
│        │                                                                     │
│        ▼                                                                     │
│   return final                                                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 한 줄 결론

현재 코드는 다음 **5가지 원칙으로 multi-process cache 일관성을 보장**하도록 짜여 있음:

1. **입력 동기화**: NCCL all_gather 로 모든 rank 가 같은 demand matrix 를 본다.
2. **결정론적 정책**: owner/evict 가 모두 deterministic tie-break.
3. **Sequential apply**: rank_planner 가 매 fetch op 마다 cache_view 즉시 mutate.
4. **All-ranks plan**: 모든 rank 가 모든 rank 의 plan 을 produce → shadow 가 byte-identical.
5. **즉시 verify**: Phase 4 가 매 layer 끝에 shadow vs archer drift 를 raise.

이 5가지가 만족되면 **cross-rank lock 없이도 일관성 유지**되고, 깨지면 Phase 4 가 즉시 죽음. 즉, 코드가 침묵하지 않고 망가지는 순간 알람이 울리는 구조.

검증 절차는 §6 의 체크리스트를 따르면 됨.
