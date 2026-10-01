# MoE Expert Cache 기작 — 전체 해설 (coslot)

> 대상: `MoE-Infinity-EP-coslot` (Controller-Owned Slot Execution)
> backend: `MoE-Infinity-EP-archer-coslot` (archer C++/CUDA)
> 작성: 2026-06-01

이 문서는 "MoE expert weight를 GPU에 캐시했다가, 부족하면 디스크/호스트에서
가져오고(fetch), 자리가 없으면 내쫓는(evict)" 전체 메커니즘이 **어느 코드에서,
어떤 순서로, 어떤 알고리즘으로** 동작하는지 빠짐없이 설명합니다.

---

## 0. 한 문단 요약

> 매 MoE layer마다, **Python Controller**가 "이번 layer에서 어떤 expert가
> 필요한지(demand)"를 모든 rank에서 모아(NCCL all_gather), GPU에 이미 있는
> 것(**hit**)과 없는 것(**miss**)으로 나눈다. miss는 어느 rank가 어느 **slot**에
> 채울지, 그리고 자리가 없으면 누구를 **evict**할지까지 **결정론적으로** 계획
> (Plan)한다. 이 Plan을 **archer C++ 엔진**에 넘기면, archer가 고정 크기 **slot
> pool**에 host→GPU 비동기 복사(H2D)로 weight를 실어 GEMM을 돌리고, layer 끝에
> Python의 "그림자(shadow) 캐시 상태"와 archer의 실제 GPU 잔류 상태가 **byte
> 단위로 일치**하는지(drift=0) 검증한다.

핵심은 **"제어(decision)는 Python이 독점, 실행(copy/compute)은 archer가 독점"**
이라는 분리입니다.

---

## 1. 코드 위치 지도 (어디에 구현되어 있나)

### 1-A. 활성 경로 (현재 coslot에서 실제로 도는 코드)

| 계층 | 파일 | 역할 |
|---|---|---|
| **모델 진입** | [models/qwen_decoder_ep.py](../moe_infinity_ep/models/qwen_decoder_ep.py), [models/qwen_ep.py](../moe_infinity_ep/models/qwen_ep.py) | decoder layer가 MoE 부분을 executor로 위임 |
| **실행 오케스트레이터** | [runtime/ep_executor.py](../moe_infinity_ep/runtime/ep_executor.py) | layer 단위 Phase 0~4 진행, NCCL a2a, archer 호출 |
| **엔진 조립/배선** | [runtime/distributed_engine.py](../moe_infinity_ep/runtime/distributed_engine.py) | archer 엔진·dispatcher·controller 생성 및 wiring, slot pool 초기화 |
| **컨트롤러 (두뇌)** | [controller/global_controller.py](../moe_infinity_ep/controller/global_controller.py) | **single source of truth.** Phase 0/1/2/4 |
| **그림자 캐시 자료구조** | [controller/slot_cache.py](../moe_infinity_ep/controller/slot_cache.py) | `RankSlotCache` / `GlobalSlotCacheView` — slot 배열 |
| **demand 수집** | [controller/demand_collector.py](../moe_infinity_ep/controller/demand_collector.py) | Phase 1.1 NCCL all_gather (token count 보존) |
| **rank 분배 정책** | [controller/owner_policy.py](../moe_infinity_ep/controller/owner_policy.py) | Phase 1.4 miss expert → 어느 rank가 담당 |
| **per-rank 계획** | [controller/rank_planner.py](../moe_infinity_ep/controller/rank_planner.py) | Phase 2 slot·victim 결정 + 즉시 shadow mutate |
| **eviction 정책** | [controller/evict_policy.py](../moe_infinity_ep/controller/evict_policy.py) | victim slot 선택 (LRU/LFU/DemandAware) |
| **archer 다리** | [controller/command_dispatcher.py](../moe_infinity_ep/controller/command_dispatcher.py) | Plan을 내 rank 것만 필터 → `submit_plan` 호출, `get_cached_experts` 조회 |
| **자료형 정의** | [controller/layer_plan.py](../moe_infinity_ep/controller/layer_plan.py) | `ExpertKey/FetchOp/HitOp/LayerPlan/Phase1Output` |
| **토큰 라우팅** | [exec/nvlink_router.py](../moe_infinity_ep/exec/nvlink_router.py) | expert→rank table, all-to-all pack/route/return |
| **(옵션) 우선순위** | [controller/priority_aggregator.py](../moe_infinity_ep/controller/priority_aggregator.py) | EAMC 예측 priority all_reduce (env로 on) |

### 1-B. archer C++/CUDA backend (실제 fetch/evict/compute)

| 파일 | 역할 |
|---|---|
| `core/parallel/expert_dispatcher.{h,cpp}` | **핵심.** slot pool, PlanQueue, direct/staging fetch, GEMM, combine |
| `core/model/model_topology.h` | `Node` — expert 1개의 host/device 포인터 + `fetch_event`/`compute_event` |
| `core/parallel/expert_module.h` | `MoEMLP`, `SetTensorsFromBlob` (slot 메모리에 weight tensor alias) |
| `core/aio/archer_tensor_index.h` | tensor_id → 디스크 위치(file_id/offset/size/shape) 매핑 |
| `core/aio/archer_prio_aio_handle.h` | 우선순위 비동기 디스크 I/O |
| `core/memory/pinned_memory_pool.h` | H2D 소스가 될 pinned host 버퍼 풀 |
| `core/python/py_archer_prefetch.cpp` | pybind — Python에 노출되는 함수 정의 |

### 1-C. Legacy / dead-code (현재 경로에서 **안 씀**, 혼동 주의)

`cache/sync.py`, `cache/view.py`, `controller/cache_state.py`,
`controller/eviction_planner.py`, `controller/placement_planner.py`,
`controller/consistency_checker.py`, `policies/*` 는 coslot 이전의
**분산형(decentralized) 아키텍처** 잔재입니다. 현재는 `controller/slot_cache.py`
+ `global_controller.py` 의 **단일 컨트롤러 경로**로 대체되었습니다.
(메모리: *single controller path* — legacy 부활 금지)

> **"expert cache가 어디 있냐?"의 한 줄 답:**
> 논리(shadow) 캐시 = [controller/slot_cache.py](../moe_infinity_ep/controller/slot_cache.py),
> 물리(GPU) 캐시 = archer `expert_dispatcher.cpp`의 **slot pool**.

---

## 2. 큰 그림 — 3계층 구조

```
┌──────────────────────────────────────────────────────────────────┐
│  Python: Controller (두뇌)        — "무엇을 어디에" 결정            │
│    GlobalCacheController                                           │
│      ├─ GlobalSlotCacheView  ← shadow cache (rank별 slot 배열)     │
│      ├─ DemandCollector      ← NCCL all_gather                     │
│      ├─ OwnerPolicy          ← miss를 rank에 분배                  │
│      ├─ RankPlanner + EvictPolicy ← slot/victim 결정              │
│      └─ CommandDispatcher    ← archer로 Plan 전달 / 상태 조회      │
└───────────────┬──────────────────────────────────────────────────┘
                │ submit_plan(hit_ops, miss_ops)   ↑ get_cached_experts()
┌───────────────▼──────────────────────────────────────────────────┐
│  Python: EPExpertExecutor (오케스트레이터) — layer 진행/NCCL a2a   │
└───────────────┬──────────────────────────────────────────────────┘
                │ pybind
┌───────────────▼──────────────────────────────────────────────────┐
│  C++/CUDA: archer ExpertDispatcher (손발)  — 실제 copy/compute     │
│    slot pool[cap+1]  ← 물리 GPU 캐시 (고정 크기 슬롯)             │
│    h2d_streams_      ← 비동기 H2D 복사                             │
│    PlanQueue         ← direct/staging fetch 스케줄                 │
│    exec stream       ← expert GEMM + combine                      │
│   (host pinned ← disk via AIO,  cudaEvent로 gating)               │
└──────────────────────────────────────────────────────────────────┘
```

**왜 이렇게 나눴나?**
모든 rank의 Controller가 **같은 입력**(all_gather된 demand)으로 **결정론적
알고리즘**을 돌리면, 서로 통신 없이도 **모든 rank가 동일한 Plan**을 만든다.
그래서 어느 rank가 어떤 토큰을 어디로 보낼지(routing)와 누가 무엇을 fetch할지가
자동으로 일치한다. archer는 그 Plan을 "그대로 실행"만 하면 된다.

---

## 3. 핵심 자료구조

### 3-A. Shadow cache — `RankSlotCache` (Python)

[slot_cache.py:54](../moe_infinity_ep/controller/slot_cache.py#L54)

- rank마다 **고정 크기 slot 배열** `slots[0..cap-1]`, 각 칸은 `(layer_id,
  expert_id)` 하나 또는 비어있음(`None`).
- `expert_to_slot`: expert → slot 역인덱스.
- `meta[slot]` (`SlotMeta`): `inserted_at`(layer 시퀀스), `last_used`(touch
  카운터), `freq`(누적 토큰 수) — eviction 정책이 점수 매길 때 읽는 값.
- **`cap`은 byte budget이 아니라 슬롯 개수**다. byte 한도는 archer의
  `cache_sizes_`가 따로 관리. (slot_cache.py:26 주석)

핵심 연산 `apply(slot, evict, insert, demand_count)`
([slot_cache.py:106](../moe_infinity_ep/controller/slot_cache.py#L106)):
한 슬롯을 **원자적으로 교체**한다. 세 가지 합법 케이스:
- `evict=A, insert=B` : A를 빼고 B를 넣음 (full replace)
- `evict=None, insert=B` : 빈 슬롯에 B를 채움 (fill)
- `evict=A, insert=None` : 순수 eviction

매 `apply`마다 불변식 I1~I4를 assert해서 shadow가 어긋나면 즉시 터진다.

`GlobalSlotCacheView` ([slot_cache.py:221](../moe_infinity_ep/controller/slot_cache.py#L221))
는 rank별 `RankSlotCache`를 모은 것. 모든 controller에서 **byte-identical하게
복제**되며, `locate(key)`로 "이 expert를 가진 가장 낮은 rank"를 찾는다.

### 3-B. 물리 캐시 — archer slot pool (C++)

`expert_dispatcher.h` / `InitSlotPool`

- `slot_pool_base_[gpu]` : `cudaMalloc`로 잡은 큰 연속 버퍼.
- `slot_device_ptr_[gpu][slot]` : 각 슬롯의 GPU 주소. 슬롯 크기 = expert 1개
  byte 크기. **`cap+1`번째 슬롯은 staging 전용 예약**.
- `slot_to_key_` / `key_to_slot_` / `cached_experts_` : 슬롯↔expert 매핑 + 잔류
  집합. `get_cached_experts(gpu)`가 이 `cached_experts_`를 그대로 반환 → Python
  의 verify 대상(ground truth).
- `slot_last_compute_event_[gpu][slot]` : 이 슬롯의 직전 GEMM 완료 event
  (staging D2D gating용).

`Node` (`model_topology.h`) = expert 1개:
`host_memory_ptr`(pinned host), `device_memory_ptr`(slot 주소),
`fetch_event`(H2D/D2D 완료), `compute_event`(GEMM 완료), `device`(DISK/CPU/CUDA).

**메모리 계층**: Disk(`.pt` weight blob) → host pinned(`PinnedMemoryPool`) →
GPU slot. demand가 생기면 disk→host(AIO)→GPU(H2D) 순으로 올라온다.

---

## 4. Phase 단위 전체 Flow (한 layer)

진입점은 [ep_executor.py `_coop_dispatch`](../moe_infinity_ep/runtime/ep_executor.py#L175)
(ep_size>1). ep_size==1은 NCCL만 빠진 같은 경로
([`_single_rank_dispatch`](../moe_infinity_ep/runtime/ep_executor.py#L146)).

```
router_logits = gate(hidden);  router_mask = topk_softmax(...)
        │
   ┌────▼─────────────────────────────────────────────── Phase 0
   │ begin_layer(layer_id)            (직전 layer는 Phase4서 이미 verify)
   ├────────────────────────────────────────────────────  Phase 1
   │ classify_and_kick_hit(layer_id, router_mask)
   │   1.1 DemandCollector.collect_with_count → per_rank[ep,E] (NCCL all_gather)
   │   1.2 union demand → 각 expert를 cache_view.locate()로 hit/miss 분류
   │   1.3 hit은 touch_use()로 meta 갱신 (LRU/LFU 신선도)
   │   1.4 OwnerPolicy.assign_rank(miss) → miss_per_rank{rank:[expert..]}
   │   ⇒ Phase1Output (hit_ops, miss_per_rank, demand, per_rank_count)
   ├────────────────────────────────────────────────────  Phase 2
   │ plan_misses(p1)
   │   for rank in 0..ep:
   │     RankPlanner.plan(rank, misses, rank_cache, demand)
   │       · demand 내림차순 정렬 (결정론)
   │       · for expert: 빈 슬롯 있으면 거기, 없으면 EvictPolicy.pick_victim()
   │       · FetchOp 만들고 ★ rank_cache.apply() 즉시 shadow mutate ★
   │   ⇒ LayerPlan (모든 rank의 fetch_ops + routing_map)
   ├──────────────────── (unified routing table 구성) ───────────────
   │ expert_to_rank = hit owner ∪ miss fetcher  (disjoint, assert)
   │ build_expert_rank_table()
   ├──────────────────── forward all-to-all ────────────────────────
   │ pack_tokens + route_tokens : 모든 토큰을 담당 rank로 1회 전송
   │ → recv_hidden[K_recv, H]
   │ set_inputs(recv_hidden, recv_mask, recv_weights)   (archer로 입력 전달)
   ├──────────────────── submit + compute (archer) ─────────────────  Phase 3
   │ command_dispatcher.submit_plan(plan, hit_ops, my_rank)
   │   → 내 rank의 hit/miss op만 필터 → local_dispatcher.submit_plan(0,...)
   │ final_local = wait_layer_done()  ← archer가 fetch+GEMM+combine 끝낼 때까지
   ├──────────────────── backward all-to-all ───────────────────────
   │ return_outputs + scatter_combine_into : 결과를 원래 토큰 위치로 복귀
   ├────────────────────────────────────────────────────  Phase 4
   │ verify_layer_end(layer_id)
   │   archer.get_cached_experts(0)  vs  shadow(this rank)  → drift==0 or RAISE
   │   cache_view.bump_all_layers()
   └─────────────────────────────────────────────────────────────────
```

### Phase별 상세

**Phase 1.1 — demand 수집**
[demand_collector.py:49](../moe_infinity_ep/controller/demand_collector.py#L49)
`collect_with_count`. 각 rank의 `router_mask`를 expert축으로 합산해
`local_count[E]`를 만들고 NCCL all_gather → `per_rank[ep_size, E]` (CPU,
int64). 모든 rank가 byte-identical한 행렬을 갖는다. 여기서
`demand[(layer,e)] = Σ_rank per_rank[:,e]` (이 layer의 글로벌 토큰 수)와
`per_rank_count`(fast-a2a split 계산용)가 나온다.
([global_controller.py:107](../moe_infinity_ep/controller/global_controller.py#L107))

**Phase 1.2 — hit/miss 분류**
demand된 expert마다 `cache_view.locate(key)`. owner가 있으면 **hit**
(`HitOp(expert, owner_rank, slot)`), 없으면 **miss**. union dedup이라 같은
expert를 여러 rank가 원해도 한 번만 fetch한다.

**Phase 1.4 — rank 분배** ([owner_policy.py](../moe_infinity_ep/controller/owner_policy.py))
miss expert를 어느 rank가 담당할지 결정. 세 정책:
- `StaticPlacement`: `rank = expert_id % ep_size` (naive, locality 보존)
- `BalancedPlacement`: 가장 적게 받은 rank에 배정 (fetch 개수 균등)
- `DemandAwareOwner`: 토큰 많은 expert부터 가장 한가한 rank에 (LPT, 트래픽 균등)

> 메모리 *owner-policy fetch-bound*: balanced는 per-layer PCIe fetch를
> 균등화하지만 decode는 weight-copy bound라 효과 제한적.

**Phase 2 — slot/victim 계획** ([rank_planner.py:42](../moe_infinity_ep/controller/rank_planner.py#L42))
rank별로 demand 내림차순 정렬 후 순차 처리:
```
for expert in ordered:
    if 이미 resident: skip
    empty = first_empty_slot()
    if empty 있음:  dst_slot=empty, victim=None
    else:           dst_slot = EvictPolicy.pick_victim(rank_cache, ...)
                    victim   = rank_cache.slots[dst_slot]
    FetchOp(expert, fetcher_rank=rank, dst_slot, victim, order)
    rank_cache.apply(dst_slot, evict=victim, insert=expert)   ← 즉시 반영
```
**즉시 apply가 핵심**: 다음 iteration의 victim 선택이 *방금 넣은 결과*를 보므로,
같은 빈 슬롯을 두 expert가 노리거나 방금 넣은 걸 바로 또 내쫓는 일이 없다
(I1/I2). `order`는 rank-local FIFO 순번으로, archer PlanQueue가 이 순서대로
처리한다.

**EvictPolicy** ([evict_policy.py](../moe_infinity_ep/controller/evict_policy.py)) victim 점수:
- LRU: `(last_used, inserted_at, slot)` 최소
- LFU: `(freq, last_used, inserted_at, slot)` 최소
- DemandAware: `(demand[expert], last_used, ...)` 최소 (가장 안 쓰일 것부터)

**Phase 3 — archer 제출** ([command_dispatcher.py:67](../moe_infinity_ep/controller/command_dispatcher.py#L67))
글로벌 LayerPlan에서 **내 rank 것만** 추려서:
- `hit_tuples = [(layer, expert, slot) for op if op.owner_rank==my_rank]`
- `miss_tuples = [(layer,expert,dst_slot,victim_layer,victim_expert,order) for op if op.fetcher_rank==my_rank]`
- order가 `0..n-1` 연속인지 검증 후 `local_dispatcher.submit_plan(0, hit_tuples, miss_tuples)`.

**Phase 4 — verify** ([global_controller.py:257](../moe_infinity_ep/controller/global_controller.py#L257))
`archer.get_cached_experts(0)`(실제 GPU 잔류) vs shadow(this rank).
`verify_against_archer`가 `(drift, missing, extra)` 반환. **drift>0이면 즉시
RAISE** (정상 동작에선 불가능, silent corruption의 최단 탐지). 그 후 모든 rank
`bump_layer()`로 `layer_seq` 전진.

---

## 5. archer 기작 — 실제 fetch/compute (C++/CUDA)

`submit_plan`을 받으면 archer가 하는 일:

### 5-A. PlanQueue & 두 가지 fetch 모드

`StartNextFetch()` 루프가 miss op를 순서대로 꺼내, 목적 슬롯 상태에 따라:

**(1) Direct Fetch** — 목적 슬롯이 **FREE**일 때 (`DoDirectFetch`):
```
1. dst = slot_device_ptr_[gpu][dst_slot]
2. cudaStreamWaitEvent(h2d_stream, 직전 compute_event)   // victim GEMM 끝 대기
3. node->device_memory_ptr = dst; SetTensorsFromBlob()   // weight tensor를 슬롯에 alias
4. CudaMemcpyAsync(dst, host_ptr, bytes, H2D, h2d_stream) // 비동기 H2D
5. cudaEventRecord(node->fetch_event, h2d_stream)         // 복사 완료 event
6. Commit()  // cached_experts 갱신, victim을 CPU로 detach
7. 슬롯 COMPUTING 표시 + ExecTask push (fetch_event를 gate로)
```

**(2) Staging Fetch** — 목적 슬롯이 아직 **COMPUTING**(직전 GEMM 진행 중)일 때
(`DoStagingFetchPark` → `CompleteStagingFetch`):
```
Park:  예약 슬롯(cap번째)으로 H2D를 먼저 흘려보냄 (직전 GEMM과 overlap)
Done:  슬롯 GEMM 끝나면 → cudaStreamWaitEvent → 값싼 D2D(staging→슬롯) → event record
```
즉 비싼 H2D(disk/host→GPU)는 직전 layer GEMM과 겹쳐 미리 끝내두고, 슬롯
overwrite는 compute가 끝났음이 event로 증명된 뒤에만 한다.
(메모리 *coslot rewrite*: PlanQueue/staging-park)

### 5-B. cudaEvent gating (덮어쓰기 안전성 = 불변식 I5)

- `h2d_stream`은 **non-blocking** 생성 → H2D가 NCCL·exec stream을 막지 않음.
- GEMM 직전 `cudaEventSynchronize(task.fetch_event)`로 weight 복사 완료를 보장.
- "hit으로 쓰는 슬롯을 같은 layer의 fetch가 덮어쓸 때"는 슬롯의
  `slot_last_compute_event`를 H2D/D2D가 기다리므로 **읽기(GEMM)가 항상 쓰기보다
  먼저 끝난다** → Python이 hit을 victim으로 골라도 안전(I5). 그래서 Phase 1에서
  hit을 pin할 필요가 없다.

### 5-C. 디스크 저장 & AIO

`ArcherTensorIndex`가 tensor_id → `(file_id, offset, size, shape, dtype)` 매핑을
들고 있고, weight는 `.pt` blob으로 디스크에 있다. demand 시 `ArcherPrioAioHandle`
이 우선순위 큐로 디스크→pinned host로 읽어온다. host에 올라온 expert는
`host_memory_ptr`를 가진 채 대기하다가 위 fetch로 GPU 슬롯에 올라간다.

### 5-D. combine

각 expert GEMM 결과는 `Partial(output, token_idx, weights)`로 모았다가
`WaitLayerDone()`에서 `(layer, expert, seq)` 순으로 정렬해 router weight로
가중합 → `final_local[K_recv, H]` 반환. 이 결정론적 combine 순서가 rank 간
수치 일치를 보장한다.

### 5-E. Python에 노출된 archer 함수 (`py_archer_prefetch.cpp`)

```python
dispatcher.init_slot_pool(cap, expert_byte_size)        # 슬롯 풀 할당 (0=auto)
dispatcher.register_expert(layer, expert, tensor_ids, jit_path)
dispatcher.set_inputs(hidden, router_mask, router_weight, is_decode)
dispatcher.submit_plan(gpu, hit_ops, miss_ops)          # ★ Plan 실행
final = dispatcher.wait_layer_done()                    # [K_recv,H]
dispatcher.get_cached_experts(gpu)                       # ★ verify용 잔류 집합
dispatcher.get_cache_stats() / get_phase_times() / get_fetch_mode_counts()
```

---

## 6. 불변식 (Invariants) — 정확성의 뼈대

[global_controller.py:14](../moe_infinity_ep/controller/global_controller.py#L14)

| | 내용 | 보장 위치 |
|---|---|---|
| **I1** | `python_shadow == archer_physical` | Phase 0 유지, **Phase 4 assert (drift==0)** |
| **I2** | shadow 변경은 controller만 (`rank_planner.apply`) | Phase 2 단일 경로 |
| **I3** | shadow 갱신은 archer를 안 기다림 (Phase 2 끝나면 shadow=layer 종료 상태) | 결정론적 Plan |
| **I4** | `len(slots)==len(meta)==cap` | `apply` 매번 검사 |
| **I5** | hit-compute 읽기가 fetch-replace 쓰기보다 항상 먼저 끝남 | archer `compute_event` wait |

**왜 drift=0이 가능한가?** 모든 rank의 Controller가 같은 all_gather 입력으로 같은
결정론적 알고리즘(정렬+순차 apply)을 돌려 **동일한 Plan**을 만들고, archer는 그
Plan을 글자 그대로 실행하므로, 끝나면 GPU 잔류 집합 = shadow가 정확히 일치한다.
어긋나면 그건 곧 버그이므로 Phase 4가 바로 터뜨린다.

---

## 7. "Python 기작" 한눈에 (책임 분리)

| 무엇 | 누가 (Python) |
|---|---|
| demand 모으기 | `DemandCollector` (NCCL all_gather) |
| hit/miss 판정 | `GlobalCacheController.classify_and_kick_hit` + `cache_view.locate` |
| miss를 rank에 배분 | `OwnerPolicy.assign_rank` |
| slot/victim 결정 | `RankPlanner.plan` + `EvictPolicy.pick_victim` |
| shadow 캐시 변경 | `RankSlotCache.apply` (controller 독점) |
| 토큰 라우팅/a2a | `exec/nvlink_router.py` + `EPExpertExecutor` |
| archer로 명령 전달 | `CommandDispatcher.submit_plan` |
| 정합성 검증 | `verify_layer_end` ↔ `get_cached_experts` |

archer(C++)는 **fetch(H2D/D2D)·GEMM·combine·잔류집합 보고**만 담당. 두 세계의
접점은 딱 두 함수: **`submit_plan`(명령)**, **`get_cached_experts`(상태 조회)**.

---

## 8. 배선(wiring) — 시작 시 한 번

[distributed_engine.py](../moe_infinity_ep/runtime/distributed_engine.py)
`_build_controller` ([:250](../moe_infinity_ep/runtime/distributed_engine.py#L250)):
`OwnerPolicy`(`MOE_EP_OWNER_POLICY`), `EvictPolicy`(`MOE_EP_EVICT_POLICY`),
`RankPlanner`, `DemandCollector`로 `GlobalCacheController`를 만든다.
`_wire_controller_components` ([:388](../moe_infinity_ep/runtime/distributed_engine.py#L388))
에서 archer dispatcher가 준비된 뒤 `CommandDispatcher`를 붙이고
`init_slot_pool(cap, 0)`로 GPU 슬롯 풀을 잡는다. archer 자체의 LFU evict는
**끄고**(`MOE_EP_DISABLE_ARCHER_EVICT=1`) 모든 캐시 결정을 controller가 독점한다
(이중 제어 루프 금지 = single controller path).

---

## 9. 관련 환경변수 (동작 토글)

| env | 기본 | 효과 |
|---|---|---|
| `MOE_EP_OWNER_POLICY` | static | miss→rank 분배 정책 |
| `MOE_EP_EVICT_POLICY` | lru | victim 선택 정책 |
| `MOE_EP_VERIFY_LAYER_END` | 1 | Phase 4 drift 검증 on/off |
| `MOE_EP_FAST_A2A` | 1 | demand 행렬로 a2a split 도출(exchange 생략) |
| `MOE_EP_DISABLE_ARCHER_EVICT` | (entry서 1) | archer 자체 evict 끔 |
| `MOE_EP_HEAVY_DEBUG` | 0 | Phase별 상세 로그 |
| `MOE_EP_ROUTING_TRACE` | 0 | layer별 plan/cache JSONL 덤프 |

---

## 부록 A — 한 layer를 한 줄씩 따라가기 (ep_size>1)

1. `gate(hidden)` → router_logits → `topk_softmax` → `router_mask`
2. `begin_layer` (Phase 0)
3. `classify_and_kick_hit`: all_gather demand → hit/miss 분류 → miss를 rank 분배
4. `plan_misses`: rank별 slot/victim 결정 + shadow 즉시 mutate → `LayerPlan`
5. `build_expert_rank_table`: hit owner ∪ miss fetcher → expert→GPU 표
6. `pack_tokens`+`route_tokens`: 모든 토큰을 담당 rank로 forward a2a → recv_hidden
7. `set_inputs(recv...)` → `command_dispatcher.submit_plan(plan, hit_ops, rank)`
8. archer: direct/staging fetch(H2D) → GEMM → combine → `wait_layer_done` → final_local
9. `return_outputs`+`scatter_combine_into`: backward a2a로 결과를 원위치 복귀
10. `verify_layer_end`: `get_cached_experts` vs shadow → drift==0 or RAISE → `bump_layer`

---

## 부록 B — 더 읽을 거리 (repo 내 문서)

- [docs/controller_flow_story.md](controller_flow_story.md) — 컨트롤러 서사적 설명
- [docs/controller_flow_verification.md](controller_flow_verification.md) — 불변식 검증 근거
- [docs/ep_offloading_flow.md](ep_offloading_flow.md) — EP offloading flow
- [moe_infinity_ep/feedback.md](../moe_infinity_ep/feedback.md) — 설계 결정 메모
