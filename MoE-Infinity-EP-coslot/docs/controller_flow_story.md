# Controller / Cache / Fetch 흐름 — 스토리 버전

> **읽기 순서**: 이 문서를 먼저 읽고, 정확한 함수/파일 위치가 필요하면 [controller_flow_verification.md](controller_flow_verification.md) 로.
>
> **시나리오 설정** (작은 예시):
> - **4개 rank** (rank 0, 1, 2, 3) — 각각 GPU 1개
> - 모델: **num_experts = 8** (expert id 0~7), **num_layers = 2** (layer 0, 1)
> - 각 rank 의 **cache capacity = 4 slots** (작게 잡음, 실제는 수백 slot)
> - 한 step 의 token = rank 마다 다른 batch, 각 token 이 expert 2개 선택 (top-2)

---

## 등장인물 정리

| 이름 | 역할 | 위치 |
|------|------|------|
| **너 (사용자)** | model.forward() 호출 | python script |
| **EPExpertExecutor** | 한 layer 의 master orchestrator | 각 rank 의 python process |
| **GlobalCacheController** | "누가 뭘 fetch 할까" 결정자 | 각 rank 의 python process |
| **GlobalSlotCacheView** | "지금 모든 rank 가 뭘 갖고 있는지" 들고 있는 장부 | controller 안에 있음 |
| **CommandDispatcher** | controller 의 결정을 archer 명령으로 번역 | 각 rank 의 python process |
| **archer** | 진짜 GPU 메모리 / PCIe / GEMM 실행자 | C++ extension |
| **NCCL** | rank 간 통신 (all_gather, all_to_all) | torch.distributed |

핵심: **GlobalCacheController, GlobalSlotCacheView, CommandDispatcher 는 rank 마다 따로 존재한다.** 즉 rank 0의 controller, rank 1의 controller, ... rank 3의 controller 가 각자 자기 python process 안에서 굴러간다. 그런데 이들은 **같은 입력을 받고 같은 코드를 돌리기 때문에 결과가 같다**.

---

## 0. 시작 — 모두 비어있는 상태

forward 첫 호출 직전, 4개 rank 의 controller 모두 같은 view 를 들고 있다:

```
GlobalSlotCacheView  (모든 rank 가 똑같이 이걸 들고 있음)
┌─────────┬────────┬────────┬────────┬────────┐
│         │ slot 0 │ slot 1 │ slot 2 │ slot 3 │
├─────────┼────────┼────────┼────────┼────────┤
│ rank 0  │   ─    │   ─    │   ─    │   ─    │
│ rank 1  │   ─    │   ─    │   ─    │   ─    │
│ rank 2  │   ─    │   ─    │   ─    │   ─    │
│ rank 3  │   ─    │   ─    │   ─    │   ─    │
└─────────┴────────┴────────┴────────┴────────┘
```

archer 의 실제 GPU 메모리도 비어있음. shadow (view) == physical (archer) 상태. OK.

---

## 1. Step 1, Layer 0 — 첫 forward

### 1.1 각 rank 의 입력 (자기 batch 의 router 결과)

각 rank 가 자기 token 을 router 에 통과시킨 결과:

```
Rank 0 의 token 들이 원하는 expert : {0, 1, 3}
Rank 1 의 token 들이 원하는 expert : {1, 2, 5}
Rank 2 의 token 들이 원하는 expert : {0, 4, 7}
Rank 3 의 token 들이 원하는 expert : {3, 5, 6}
```

이 시점에서 **rank 0 은 rank 1, 2, 3 이 뭘 원하는지 모른다**. 자기 batch 만 봤기 때문.

### 1.2 Phase 1.1 — 모두가 같은 그림을 본다 (NCCL all_gather)

`DemandCollector.collect_with_count()` 가 NCCL all_gather 한 번 호출.

> NCCL all_gather: "내 정보 너희에게 다 줄게, 너희 것도 다 받을게" — 모든 rank 가 똑같은 결과를 받는다는 게 핵심.

호출 끝나면 **모든 rank** 가 같은 `[4 rank × 8 expert]` 행렬을 받음:

```
per_rank[rank, expert] = 그 rank 의 token 중 그 expert 를 원하는 token 수

per_rank =
        e0   e1   e2   e3   e4   e5   e6   e7
rank 0 [ 5,   3,   0,   2,   0,   0,   0,   0 ]
rank 1 [ 0,   4,   7,   0,   0,   1,   0,   0 ]
rank 2 [ 2,   0,   0,   0,   6,   0,   0,   3 ]
rank 3 [ 0,   0,   0,   8,   0,   2,   4,   0 ]
```

→ rank 0 도 이 행렬을 받고, rank 3 도 정확히 같은 행렬을 받는다.

### 1.3 Phase 1.2 — Hit/Miss 분류

각 rank 의 controller 가 **이 행렬을 자기 view 와 비교**:

```
union demanded experts = 행렬 합쳐서 nonzero = {0, 1, 2, 3, 4, 5, 6, 7}  (전부 demand 됨)

각 expert 가 cache_view 어디 있나?
  e0 → view 어디에도 없음 → MISS
  e1 → 없음 → MISS
  e2 → 없음 → MISS
  ... (지금은 다 비어있으므로) 전부 MISS
```

**모든 rank 가 같은 결론에 도달**: `hit_set = ∅`, `miss_set = {(0,0), (0,1), ..., (0,7)}` (layer 0 의 expert 8개 전부).

> 여기서 `(0, 3)` 같은 표기는 (layer_id=0, expert_id=3) 의미.

### 1.4 Phase 1.4 — Owner 분배 (어느 rank 가 책임지나?)

`owner_policy.assign_rank()` 가 miss 들을 rank 에 분배. `static_placement` (= `expert % ep_size`) 라고 가정:

```
e0 → rank 0  (0 % 4 = 0)
e1 → rank 1  (1 % 4 = 1)
e2 → rank 2  (2 % 4 = 2)
e3 → rank 3  (3 % 4 = 3)
e4 → rank 0  (4 % 4 = 0)
e5 → rank 1
e6 → rank 2
e7 → rank 3
```

결과 **모든 rank 가 같이 계산하므로 모두 같은 결론**:

```
miss_per_rank = {
  0: [(0,0), (0,4)],
  1: [(0,1), (0,5)],
  2: [(0,2), (0,6)],
  3: [(0,3), (0,7)],
}
```

→ rank 0 의 controller 도 "내가 e0, e4 가져온다" 알고, "rank 3 은 e3, e7 가져온다" 도 안다. **모두 알고 있음**.

### 1.5 Phase 2 — Per-rank 계획 (어느 slot 에 넣나?)

이제 controller 가 **모든 rank 의 plan 을 자기가 직접 produce**. 즉 rank 0 의 controller 도 "rank 3 이 어떻게 slot 배치할지" 까지 똑같이 계산한다.

```python
for r in [0, 1, 2, 3]:
    rank_planner.plan(rank=r, miss_experts=miss_per_rank[r], 
                      rank_cache=cache_view.per_rank[r], demand=...)
```

**rank 0 의 plan** (rank 0의 controller 가 자기 view 의 rank 0 부분을 보고):
- 들어올 expert: [(0,0), (0,4)] (demand 정렬: e0 의 token=7, e4 의 token=6 이라 e0 먼저)
- e0 fetch: rank 0 의 slot 다 비어있음 → slot 0 채움. victim 없음.
  → `FetchOp(expert=(0,0), fetcher_rank=0, dst_slot=0, victim=None)`
  → 즉시 `cache_view.per_rank[0].apply(slot=0, evict=None, insert=(0,0))`
- e4 fetch: 이제 slot 1, 2, 3 비어있음 → slot 1 채움. victim 없음.
  → `FetchOp(expert=(0,4), fetcher_rank=0, dst_slot=1, victim=None)`
  → 즉시 apply

**rank 3 의 plan** (rank 0 의 controller 가 rank 3 자리에 대해 계산):
- 들어올 expert: [(0,3), (0,7)]
- 같은 로직으로 slot 0 에 e3, slot 1 에 e7 install.
- `cache_view.per_rank[3].apply(...)` 두 번.

**모든 rank 가 모든 rank 의 plan 을 같이 실행했으므로**, Phase 2 끝난 직후 모든 rank 의 view 는 다음과 같이 **byte-identical**:

```
GlobalSlotCacheView  (rank 0~3 모두 똑같이 이걸 봄)
┌─────────┬────────┬────────┬────────┬────────┐
│         │ slot 0 │ slot 1 │ slot 2 │ slot 3 │
├─────────┼────────┼────────┼────────┼────────┤
│ rank 0  │ (0,0)  │ (0,4)  │   ─    │   ─    │
│ rank 1  │ (0,1)  │ (0,5)  │   ─    │   ─    │
│ rank 2  │ (0,2)  │ (0,6)  │   ─    │   ─    │
│ rank 3  │ (0,3)  │ (0,7)  │   ─    │   ─    │
└─────────┴────────┴────────┴────────┴────────┘
```

⚠️ **여기서 중요한 점**: 이건 아직 **shadow** (의도) 일 뿐. archer 의 진짜 GPU 메모리에는 아직 expert weight 가 안 들어가 있음. 단순히 controller 의 장부가 미리 갱신된 것.

### 1.6 Phase 3d — 실제 fetch 명령 (자기 rank 것만)

이제 각 rank 의 `CommandDispatcher` 가 **자기 rank 의 FetchOp 만 골라서** archer 에 명령:

```
Rank 0 의 dispatcher:
  for op in plan.fetch_ops:
    if op.fetcher_rank == 0:        # 자기 것만
      explicit_replace_async(...)   # archer h2d_stream 에 PCIe 시작
  → e0, e4 두 개 archer 에 명령

Rank 1 의 dispatcher:  e1, e5 명령
Rank 2 의 dispatcher:  e2, e6 명령
Rank 3 의 dispatcher:  e3, e7 명령
```

→ **모든 rank 가 자기 GPU 의 PCIe 채널만 사용**. 총 8 expert가 4 PCIe 채널로 병렬 fetch. 효율적.

⚠️ **shadow vs physical 차이 발생 시점**: 이 명령은 non-blocking. archer 는 "받았어, 백그라운드에서 할게" 하고 return. 그래서 이 순간:

- **shadow** (cache_view): e0~e7 이미 다 install 된 것으로 표기
- **physical** (archer): 아직 PCIe 진행 중, 실제 GPU 메모리는 거의 비어있음

이 gap 은 **Phase 3g (miss GEMM)** 에서 archer 가 `fetch_event` 대기로 알아서 닫는다. controller 가 신경 쓸 필요 없음.

### 1.7 Phase 3a/3e/3b/3g — Token routing + GEMM

이건 controller/cache 입장 에서는 read-only 라 간단히만:

- Phase 3a: hit 토큰 NCCL all_to_all (이 layer 는 hit 0 이라 skip)
- Phase 3e: miss 토큰 NCCL all_to_all — rank 0 의 e0/e4 가는 token 들이 rank 0 GPU 로 모임
- Phase 3b: hit GEMM (skip)
- Phase 3g: rank 0 가 e0/e4 의 GEMM 실행. archer 가 fetch 완료 자동 대기.
- Phase 3c/3h: 결과 다시 NCCL all_to_all 역방향 → 원래 token 자리로

### 1.8 Phase 4 — 검증

각 rank 가 archer 에 "지금 너 GPU 에 뭐 있어?" 물어봄:

```python
physical_rank_0 = local_dispatcher.get_cached_experts(0)
                = {(0,0), (0,4)}                              ← archer 실제 상태

shadow_rank_0   = cache_view.per_rank[0].occupied_keys()
                = {(0,0), (0,4)}                              ← controller 장부

drift = shadow XOR physical = ∅  →  drift_count = 0  ✓
```

만약 여기서 drift > 0 이면 즉시 `RuntimeError` raise. 코드가 침묵하지 않고 죽음.

마지막에 `bump_all_layers()` → 모든 rank 의 `layer_seq` += 1.

---

## 2. Step 1, Layer 1 — 같은 패턴 반복

layer 1 도 똑같은 과정. router 가 layer 1 의 expert (e0~e7) 중 뭐 원하는지 결정, 모든 게 비어있으므로 다 miss, 분배, fetch, GEMM, verify. 끝나면:

```
GlobalSlotCacheView  (layer 1 의 expert 들이 slot 2, 3 채움)
┌─────────┬────────┬────────┬────────┬────────┐
│         │ slot 0 │ slot 1 │ slot 2 │ slot 3 │
├─────────┼────────┼────────┼────────┼────────┤
│ rank 0  │ (0,0)  │ (0,4)  │ (1,0)  │ (1,4)  │
│ rank 1  │ (0,1)  │ (0,5)  │ (1,1)  │ (1,5)  │
│ rank 2  │ (0,2)  │ (0,6)  │ (1,2)  │ (1,6)  │
│ rank 3  │ (0,3)  │ (0,7)  │ (1,3)  │ (1,7)  │
└─────────┴────────┴────────┴────────┴────────┘
```

이제 4 slots 다 차있음. step 2 부터는 eviction 발생.

---

## 3. Step 2, Layer 0 — Hit 등장 + 부분 Miss

### 3.1 Router 결과

이번에는 token 들이 부분적으로 같은 expert 를 다시 원함:

```
Rank 0 의 token: expert {0, 2, 5}  ← e0 은 cache 에 있음, e2/e5 는 다른 rank
Rank 1 의 token: expert {1, 3, 7}
Rank 2 의 token: expert {4, 6, 0}
Rank 3 의 token: expert {3, 5, 1}
```

### 3.2 Phase 1 — all_gather + 분류

```
union demanded = {0, 1, 2, 3, 4, 5, 6, 7}  (전부)

각 expert 가 cache_view 어디 있나?
  e0 → rank 0 의 slot 0 에 있음 → HIT (owner_rank=0, slot=0)
  e1 → rank 1 의 slot 0 → HIT
  e2 → rank 2 의 slot 0 → HIT
  e3 → rank 3 의 slot 0 → HIT
  e4 → rank 0 의 slot 1 → HIT
  e5 → rank 1 의 slot 1 → HIT
  e6 → rank 2 의 slot 1 → HIT
  e7 → rank 3 의 slot 1 → HIT
```

**layer 0 의 expert 들은 전부 cache 에 있음**. 그런데 우리가 원하는 건 layer 0 의 expert 라서... 잠깐, 이건 hit/miss 가 (layer_id, expert_id) 단위라는 걸 다시 강조해야 함.

layer 0 forward 중이면 demand 도 `(0, *)`. cache 에 있는 것도 `(0, 0)`, `(0, 1)`, ..., `(0, 7)` — 다 layer 0 거. 즉 **이번 layer 0 forward 는 100% hit**.

Phase 1.3 에서 hit touch:
```python
for op in hit_ops:
    cache_view.per_rank[op.owner_rank].touch_use(op.expert, demand_count)
    # → meta.last_used = current touch_counter (+1)
    # → meta.freq += demand_count
```

→ slot 자체는 그대로, **meta 만 갱신**. LRU/LFU 점수 update.

### 3.3 Phase 2 — Miss 가 없으니 plan 비어있음

```
miss_per_rank = {0: [], 1: [], 2: [], 3: []}
plan.fetch_ops = []
```

→ Phase 3d (fetch launch) 도 no-op. PCIe 안 씀. 깔끔.

### 3.4 Phase 3a — Hit routing all_to_all

이번엔 hit 만 있으니 hit a2a 만 실행. 예를 들어:

```
Rank 0 의 token 중 e2 원하는 애들 → rank 2 로 보냄 (e2 는 rank 2 의 slot 0)
Rank 0 의 token 중 e0 원하는 애들 → 자기 rank 0 가 처리
Rank 0 의 token 중 e5 원하는 애들 → rank 1 로 보냄
```

→ 모든 rank 가 자기 cache 에 있는 expert 의 GEMM 만 실행. PCIe 부담 0.

### 3.5 Phase 4 — Verify

shadow 안 바뀜 (touch_use 는 meta 만 변경, slots 변화 X). archer 도 안 바뀜. drift=0.

---

## 4. Step 3, Layer 0 — 드디어 Eviction 발생

### 4.1 Router 결과

이번엔 새로운 expert 가 등장:

```
Rank 0: {0, 1, 8}  ← e8 은 num_experts=8 이라 없... 잠깐 0~7 까지였지.
```

다시. expert id 는 0~7 까지인데, 다른 layer 의 expert 도 있을 수 있음. 정확하게는 (layer_id, expert_id) 페어가 cache 의 unit. 그래서 시나리오를 바꿔서 — **layer 0 forward 중인데 우리가 layer 0 의 expert 0, 1 만 다시 원하고 cap=4 라 eviction 안 일어남**. eviction 보여주려면 cap 을 줄여야 하니, **cap=2 시나리오** 로 재설정:

```
cache_view (cap=2, step 2 끝난 상태):
┌─────────┬────────┬────────┐
│         │ slot 0 │ slot 1 │
├─────────┼────────┼────────┤
│ rank 0  │ (0,0)  │ (0,4)  │
│ rank 1  │ (0,1)  │ (0,5)  │
│ rank 2  │ (0,2)  │ (0,6)  │
│ rank 3  │ (0,3)  │ (0,7)  │
└─────────┴────────┴────────┘
```

(layer 1 처리 시점에 cap 부족해서 layer 0 일부 evict 됐다고 가정).

step 3, layer 0 시작. router 결과:

```
Rank 0: {0, 1}  ← e0 hit (slot 0), e1 다른 rank
Rank 1: {0, 5}  ← e5 hit (rank 1 slot 1), e0 다른 rank
Rank 2: {1, 6}  ← e6 hit (rank 2 slot 1), e1 다른 rank
Rank 3: {2, 3}  ← e3 hit (rank 3 slot 0), e2 다른 rank
```

### 4.2 Phase 1 — 분류

```
per_rank 행렬 all_gather → 모두가 같이 봄.

union demanded = {0, 1, 2, 3, 5, 6}

분류:
  e0 → rank 0 slot 0 → HIT
  e1 → rank 1 slot 0 → HIT  (rank 1 의 cache 에 있다)
  e2 → rank 2 slot 0 → HIT
  e3 → rank 3 slot 0 → HIT
  e5 → rank 1 slot 1 → HIT
  e6 → rank 2 slot 1 → HIT

→ MISS 없음. 또 100% hit.
```

이 step 에도 fetch 없음. cache 효과 톡톡히 보는 중.

### 4.3 NCCL hit routing

각 rank 의 token 들이 적절한 rank 로 흘러감:
- rank 0 의 e1 원하는 token → rank 1 으로
- rank 1 의 e0 원하는 token → rank 0 으로
- rank 2 의 e1 원하는 token → rank 1 으로
- rank 3 의 e2 원하는 token → rank 2 로
- 등등

→ rank 1 의 e1 가 rank 0, rank 2 양쪽 token 다 처리. **expert 1개가 여러 rank 의 token 을 모아 한 번에 GEMM**.

---

## 5. Step 4, Layer 0 — 진짜 Eviction

### 5.1 Router 결과

```
Rank 0: {0, 9, 10}  ← e9, e10 처음 등장 (다른 layer 의 expert)
```

음 이 예시도 layer/expert id 같이 가야하니, layer 1 처리 중이라고 하자. **step 4, layer 1 forward**, 그리고 cap=2 라 cache 에 layer 0 expert 만 들어있음:

```
cache_view (cap=2):
┌─────────┬────────┬────────┐
│         │ slot 0 │ slot 1 │
├─────────┼────────┼────────┤
│ rank 0  │ (0,0)  │ (0,4)  │
│ rank 1  │ (0,1)  │ (0,5)  │
│ rank 2  │ (0,2)  │ (0,6)  │
│ rank 3  │ (0,3)  │ (0,7)  │
└─────────┴────────┴────────┘
```

step 4 layer 1 forward. router 결과:

```
Rank 0: layer 1 의 expert {0, 1} 원함  ← (1,0), (1,1)
Rank 1: layer 1 의 expert {2, 3} 원함
Rank 2: layer 1 의 expert {4, 5} 원함
Rank 3: layer 1 의 expert {6, 7} 원함
```

### 5.2 Phase 1 — 전부 MISS

```
union demanded (layer 1) = {(1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7)}

분류: 전부 cache 에 없음. 전부 MISS.

owner_policy.assign_rank (static_placement):
  miss_per_rank = {
    0: [(1,0), (1,4)],
    1: [(1,1), (1,5)],
    2: [(1,2), (1,6)],
    3: [(1,3), (1,7)],
  }
```

### 5.3 Phase 2 — Eviction 시작

이제 각 rank 의 slots 다 차있음 (cap=2, 2개 차있음). **rank 0 의 plan** 보면:

```
Step 5 (demand 정렬): [(1,0) token=많음, (1,4) token=적음]
# 시연을 위해 (1,0) 이 demand 더 크다고 가정

iteration 1: (1,0) install
  first_empty_slot() → None (slot 0,1 차있음)
  evict_policy.pick_victim(rank_cache=rank0, incoming=(1,0))
    LRU: meta.last_used 최소
      slot 0 의 (0,0) meta: last_used=20 (step 3 의 hit touch 때 갱신됨)
      slot 1 의 (0,4) meta: last_used=5  (step 2 이후 안 쓰임)
    → slot 1 victim, (0,4) 쫓아냄
  FetchOp(expert=(1,0), fetcher_rank=0, dst_slot=1, victim=(0,4))
  rank_cache.apply(slot=1, evict=(0,4), insert=(1,0))
  → 즉시 view 갱신:
      slots[1] = (1,0)
      meta[1] = SlotMeta(expert=(1,0), inserted_at=4, last_used=now, freq=demand)

iteration 2: (1,4) install
  first_empty_slot() → None (여전히 다 차있음)
  pick_victim:
    slot 0 의 (0,0) meta: last_used=20
    slot 1 의 (1,0) meta: last_used=방금 (touch_counter 막 +1 됨)
    → slot 0 victim, (0,0) 쫓아냄  
    (방금 install 된 (1,0) 은 last_used 가 가장 크므로 victim 안 됨 — 자연 보호)
  FetchOp(expert=(1,4), fetcher_rank=0, dst_slot=0, victim=(0,0))
  apply 즉시
```

이 두 iteration 이 **sequential & immediate apply** 의 핵심:
1. 첫 fetch 가 slot 1 차지하자마자 view 갱신.
2. 두 번째 iter 가 view 보면 slot 1 의 (1,0) 은 방금 들어왔다고 보임 (last_used 최대).
3. 그래서 자연스럽게 slot 0 이 victim 으로 선정.
4. **만약 sequential 이 아니라 두 fetch 동시에 결정했다면 둘 다 slot 1 같은 victim 노려서 충돌**.

rank 0 의 plan 끝난 후 rank 0 의 view 슬라이스:
```
rank 0: [ (1,4) , (1,0) ]   ← slot 0 evicted to (1,4), slot 1 evicted to (1,0)
```

같은 방식으로 rank 1, 2, 3 도 controller 가 동시에 계산. Phase 2 끝난 후:

```
GlobalSlotCacheView  (모든 rank 동일)
┌─────────┬────────┬────────┐
│         │ slot 0 │ slot 1 │
├─────────┼────────┼────────┤
│ rank 0  │ (1,4)  │ (1,0)  │
│ rank 1  │ (1,5)  │ (1,1)  │
│ rank 2  │ (1,6)  │ (1,2)  │
│ rank 3  │ (1,7)  │ (1,3)  │
└─────────┴────────┴────────┘

plan.fetch_ops = [
  FetchOp((1,0), fetcher=0, dst_slot=1, victim=(0,4)),
  FetchOp((1,4), fetcher=0, dst_slot=0, victim=(0,0)),
  FetchOp((1,1), fetcher=1, dst_slot=1, victim=(0,5)),
  FetchOp((1,5), fetcher=1, dst_slot=0, victim=(0,1)),
  ... (rank 2, 3 동일 패턴)
]
```

### 5.4 Phase 3d — 자기 rank 만 archer 명령

```
Rank 0 의 dispatcher:
  for op in plan.fetch_ops:
    if op.fetcher_rank == 0:
      explicit_replace_async(0, v_layer=0, v_expert=4, new_layer=1, new_expert=0)
      # → archer 내부:
      #   cached_experts_.erase((0,4))
      #   cached_experts_.insert((1,0))
      #   cudaMemcpyAsync(GPU slot 1 의 buffer, host pinned of (1,0))
      explicit_replace_async(0, v_layer=0, v_expert=0, new_layer=1, new_expert=4)

Rank 1 의 dispatcher: 자기 거 (1,1), (1,5) 명령
Rank 2 의 dispatcher: (1,2), (1,6)
Rank 3 의 dispatcher: (1,3), (1,7)
```

**4 rank 가 동시에 자기 PCIe 채널로 fetch**. 8 expert 가 4 채널 병렬 → 이론상 시간 절반.

### 5.5 Phase 4 — Verify

```
Rank 0 의 controller 가 묻기:
  archer.get_cached_experts(0) → {(1,0), (1,4)}    ← archer 실제 상태
  shadow.per_rank[0].occupied_keys() → {(1,0), (1,4)}   ← controller 장부
  drift = 0  ✓
```

만약 여기서 예컨대 archer 가 `explicit_replace_async` 중 (1,4) fetch 실패했다면:
```
archer 실제: {(1,0)}                    ← (1,4) 못 들어옴
shadow:      {(1,0), (1,4)}              ← controller 는 갔다고 믿음
missing = shadow - archer = {(1,4)}      ← drift!
→ RuntimeError raise
```

→ 즉시 멈춤. silent corruption 방지.

---

## 6. 왜 이 구조가 lock 없이 동작하는가

핵심을 다시 정리하면:

### "같은 코드 + 같은 입력 = 같은 출력" 원칙

```
                  rank 0           rank 1           rank 2           rank 3
                    │                │                │                │
input  (NCCL ─────  ✓ 같은 행렬 ──── ✓ 같은 행렬 ──── ✓ 같은 행렬 ──── ✓ 같은 행렬
        all_gather)
                    │                │                │                │
hit/miss           ✓ 같은 분류       ✓ 같은 분류       ✓ 같은 분류       ✓ 같은 분류
                    │                │                │                │
owner.assign       ✓ 같은 분배       ✓ 같은 분배       ✓ 같은 분배       ✓ 같은 분배
(deterministic)
                    │                │                │                │
rank_planner       ✓ 모두가 모든     ✓ 모두가 모든     ✓ 모두가 모든     ✓ 모두가 모든
(for r in 0..3:     rank plan        rank plan        rank plan        rank plan
   plan)            을 같이 계산     을 같이 계산     을 같이 계산     을 같이 계산
                    │                │                │                │
shadow mutate      ✓ view byte-      ✓ view byte-     ✓ view byte-     ✓ view byte-
                    identical        identical        identical        identical
                    │                │                │                │
                ┌───┴───┐        ┌───┴───┐        ┌───┴───┐        ┌───┴───┐
                │ 자기  │        │ 자기  │        │ 자기  │        │ 자기  │
                │ rank  │        │ rank  │        │ rank  │        │ rank  │
                │ fetch │        │ fetch │        │ fetch │        │ fetch │
                │ 만    │        │ 만    │        │ 만    │        │ 만    │
                │ 발행  │        │ 발행  │        │ 발행  │        │ 발행  │
                └───────┘        └───────┘        └───────┘        └───────┘
                  (자기 archer)    (자기 archer)    (자기 archer)    (자기 archer)
```

**rank 간 lock 이 필요 없는 이유**: 어차피 다 같은 결론에 도달하니 누가 먼저 결정해도 결과 같음. NCCL 동기화는 입력 줄 맞추기에만 쓰고, "누가 fetch 할지" 결정에는 통신 0.

---

## 7. 잘 동작하는지 한 눈에 보는 방법

`MOE_EP_HEAVY_DEBUG=1` 켜고 한 step 돌리면 각 rank 가 다음 로그를 찍음:

```
[ctrl rank=0] L0 P1: demanded=8 hits=0 miss_dedup=8 miss_per_rank={0: 2, 1: 2, 2: 2, 3: 2}
[ctrl rank=1] L0 P1: demanded=8 hits=0 miss_dedup=8 miss_per_rank={0: 2, 1: 2, 2: 2, 3: 2}
[ctrl rank=2] L0 P1: demanded=8 hits=0 miss_dedup=8 miss_per_rank={0: 2, 1: 2, 2: 2, 3: 2}
[ctrl rank=3] L0 P1: demanded=8 hits=0 miss_dedup=8 miss_per_rank={0: 2, 1: 2, 2: 2, 3: 2}
```

★ **`demanded`, `hits`, `miss_dedup`, `miss_per_rank` 가 4 rank 에서 똑같다** → 결정성 OK.

만약 다르면:
- 다 다르면: `DemandCollector` 의 NCCL 이상 또는 router 결과가 device dependent
- `demanded` 만 같고 `hits` 다르면: cache_view 가 어긋남 (이전 step 에서 누군가 view mutate 빼먹음)
- `miss_per_rank` 만 다르면: owner_policy 가 deterministic 아님

```
[ctrl rank=0] L0 P2: total_fetch=8 my_fetch=2 shadow_after=8
[ctrl rank=1] L0 P2: total_fetch=8 my_fetch=2 shadow_after=8
[ctrl rank=2] L0 P2: total_fetch=8 my_fetch=2 shadow_after=8
[ctrl rank=3] L0 P2: total_fetch=8 my_fetch=2 shadow_after=8
```

★ **`total_fetch`, `shadow_after` 가 같음** (my_fetch 는 자기 분량이라 다를 수 있음). plan_misses 가 모든 rank 같은 결과 produce 했다는 증거.

```
[dispatch rank=0] L0 fetches=2 fetch_fail=0 evict_fail=0
[dispatch rank=1] L0 fetches=2 fetch_fail=0 evict_fail=0
...
```

★ fetch failure 0.

Phase 4 에서 drift 안 뜸 → 끝. 의도대로.

---

## 8. 한 줄 요약

> **"4개 rank 가 NCCL 로 입력 한 번 맞추고, 각자 같은 알고리즘 굴려서 같은 결론에 도달한 다음, 자기 GPU 의 PCIe 만 써서 fetch 한다. 모두가 같은 장부를 들고 있으니 합의가 필요 없고, 매 layer 끝에 장부와 진짜를 대조해 한 곳이라도 어긋나면 즉시 죽는다."**

이게 의도이고, 코드는 이걸 강제하도록 짜여 있음.
