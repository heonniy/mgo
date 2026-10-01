# MoE-Infinity-EP: EP + Offloading 전체 Flow (time-step 상세)

> 한 layer의 forward를 time-step 순으로, **Python(EP executor) / Controller / Archer** 3주체와
> **h2d / exec / NCCL** 3스트림, 그리고 PCIe·NCCL 작용까지 phase별로 설명한다.
> 2026-05-28 실제 코드(`ep_executor._coop_dispatch`, `global_controller`, archer `ExpertDispatcher`) 기준.
> 관련 기본값/결정은 동일 디렉터리의 다른 문서 및 메모리(`project_ep_config_decisions`) 참조.

---

## 0. 등장인물 (3주체 + 3스트림)

| 주체 | 역할 | 사는 곳 |
|---|---|---|
| **Python / EP executor** (`ep_executor._coop_dispatch`) | 한 layer의 오케스트레이터. 토큰 packing, NCCL 호출, archer에 enqueue, 결과 결합 | 각 rank의 Python process |
| **Controller** (`GlobalCacheController`) | "누가 무엇을 갖고/가져올지" 결정. cache_view(shadow)가 단일 진실 | 각 rank Python (rank별 동일 결정) |
| **Archer** (`ExpertDispatcher`, C++) | 실제 PCIe fetch + GPU GEMM 실행. slot pool 메모리 관리 | 각 rank의 C++ 백엔드 + worker 스레드 |

| 스트림 | 하는 일 | 왜 분리 |
|---|---|---|
| **h2d_stream** (GPU당 1개) | host→GPU PCIe 복사 (expert weight fetch) | 느린 PCIe를 GEMM·NCCL과 **겹치려고** |
| **exec_stream** (GPU당 1개) | expert GEMM (SwiGLU) | fetch 끝난 weight만 읽도록 event로 gate |
| **NCCL stream** | all_gather(demand) + all_to_all(token routing) | rank 간 통신 |

스트림끼리는 **cudaEvent**로 의존성을 건다 (CPU는 안 막힘):
- `fetch_event`: h2d → exec (fetch 끝나야 GEMM 시작)
- `slot_last_compute_event`: exec → h2d (이전 GEMM 끝나야 그 slot 덮어씀)

---

## 1. 시작 전 — 1회 셋업 (`_wire_controller_components`)

```
t = setup:
  [Python] 모델 build (각 rank가 자기 shard 로드, NUMA 복제)
  [Python] CommandDispatcher 생성 → archer API 감지 (effective=replace_async)
  [Archer] init_slot_pool(cap, 0):
     - GPU당 cudaMalloc( cap × expert_byte + 1 × expert_byte )  ← resident slot cap개 + staging 1개, 고정 주소
     - slot마다 slot_last_compute_event 생성 (never-recorded → 첫 사용 시 no-op)
  [Controller] cache_view (per-rank slot 배열 = shadow) 비어있음
  [Archer] GPUFetchFunc / GPUExecFunc 스레드 가동, input_queue_/exec_queue_ 대기
```

이 시점: 모든 GPU 캐시 비어있음(cached_experts_ = ∅). h2d/exec 스트림 + NCCL group 준비 완료.

---

## 2. 한 layer의 forward — time-step별

예시: **4 rank(EP=4), cap=8, Qwen3-30B(128 expert, top-8)**. router가 막 돌아서 `router_mask`(이 rank의 토큰→expert)가 나온 직후.

### ▶ Phase 0 — begin_layer
```
t0  [Python→Controller] controller.begin_layer(layer_id)
       현재 layer 설정만. (가벼움)
```

### ▶ Phase 1 — classify + demand 수집 (첫 NCCL)
```
t1  [Controller] classify_and_kick_hit(layer_id, router_mask):
     ├─ (1.1) demand_collector.collect_with_count():
     │     [NCCL all_gather] ← ★첫 collective★
     │     각 rank가 "내 토큰이 expert e를 몇 개 원하는지"([num_experts]) 를 교환
     │     → per_rank[ep_size, num_experts] (CPU). 모든 rank가 동일 행렬 보유.
     ├─ union demand 계산, 각 (layer,e)를 cache_view.locate() 로 분류:
     │     resident면 HIT(owner rank 기록), 없으면 MISS
     ├─ hit들 touch (LRU 최신화)
     └─ owner_policy가 MISS들을 fetcher rank에 배정 (miss_per_rank)
     반환: routing_map_hit, miss_per_rank, demand, per_rank_count
t1' [Python] _prl = per_rank_count.tolist()   ← fast-a2a용 (sync 없음, 이미 CPU)
```
> 왜 all_gather? 각 rank는 자기 토큰만 안다. "전역으로 누가 뭘 원하나"를 알아야 hit/miss와 routing을 모든 rank가 **동일하게** 판단 → view 일관성.

### ▶ Phase 3a — hit 토큰 a2a 발사 (NCCL)
```
t2  [Python] hit experts에 대해:
     ├─ make_filtered_router_mask(hit) + build_expert_rank_table(hit_routing)
     ├─ derive_split_counts(_prl, hit_routing) → hit_send/recv_counts
     │     ★fast-a2a #1: exchange_counts collective + .cpu() sync 생략★
     ├─ pack_tokens(): 토큰을 dst rank 순으로 정렬, send_hidden 구성 (default stream, GPU)
     └─ route_tokens():
           [NCCL all_to_all] ← hidden|expert|weights를 byte-fuse해 ★1번★ collective
           ★fast-a2a #2: 3→1★
           → hit_recv (이 rank가 실행 책임진 hit expert로 모인 토큰들)
```
> hit은 이미 GPU에 resident이므로 fetch 불필요 → 토큰만 owner rank로 보냄.

### ▶ Phase 2 — miss 계획 (Controller, CPU — NCCL과 겹침)
```
t3  [Controller] plan_misses(p1):
     rank_planner가 각 rank의 miss를 LRU 순서로 shadow에 sequential apply
     → fetch_ops[ {expert, fetcher_rank, dst_slot, victim_expert} ], routing_map_miss
     이때 cache_view(shadow)를 낙관적으로 갱신 (victim out, new in)
```
> t2의 hit a2a(NCCL stream)가 도는 **동안** 이 CPU 계획이 진행 → 통신/계획 overlap.

### ▶ Phase 3b — hit GEMM 먼저 (Option A, slot 안전의 핵심)
```
t4  [Python] _archer_batch_compute(hit_recv...):
     ├─ set_inputs(hit_hidden, router_mask, weights) → archer 공유상태 세팅
     ├─ 각 unique hit expert: enqueue_expert(layer,e)
     │     [Archer.Enqueue] resident(cache_hit) → exec_queue_ 로 직행 (fetch 안 함)
     ├─ notify_fetch_start()
     └─ wait_expert()  ← blocking
           [Archer.GPUExecFunc, exec_stream]:
             cudaStreamWaitEvent(fetch_event)  (hit은 이미 fetch됨 → no-op)
             GEMM(SwiGLU) 실행
             RecordComputeEvent + slot_last_compute_event[slot] record  ★
             OutputFunc: routing weight 곱 + 토큰 자리로 scatter → output_queue
       → hit_local_output (clone)
```
> ★왜 hit을 miss fetch보다 먼저?★ hit이 쓰는 slot을 나중에 miss가 재사용(덮어쓰기)할 수 있는데,
> 여기서 hit GEMM의 `slot_last_compute_event`를 먼저 기록해두면, 뒤의 miss fetch가 그 event를 기다려
> **GEMM 끝나기 전엔 안 덮어쓴다** → Round-7 hit-victim 레이스 방지.

### ▶ Phase 3e — miss 토큰 a2a (NCCL)
```
t5  [Python] miss experts에 대해 (3a와 동일 구조):
     derive_split_counts(miss) → pack_tokens → route_tokens
       [NCCL all_to_all] miss 토큰을 fetcher rank로 → miss_recv
```

### ▶ Phase 3d+3g — miss fetch + compute (오프로딩 심장부)
`_pipelined_miss_compute`. 내 rank가 책임진 miss = `my_ops`.

**경우 A: cap ≥ demand (fast path, 균형 배치의 보통)**
```
t6  [Python] cd.launch_fetch_ops(my_ops):  각 op마다
     [Archer.explicit_replace_async(gpu, victim, new)]:
       ├─ cache_mutex 안(동기): dst_slot 결정, cached_experts_/key_to_slot_/
       │   slot_to_key_/device flag 갱신 (ghost 상태 없음)
       └─ h2d_stream 위(비동기):
            cudaStreamWaitEvent(slot_last_compute_event[dst_slot]) ← 이전 점유자 GEMM 대기
            CudaMemcpyAsync H2D  host→slot  ★PCIe fetch★
            cudaEventRecord(fetch_event)
     → cap개 H2D가 h2d_stream에 연달아 쌓임
t7  [Python] _archer_batch_compute(all miss): 각 miss expert enqueue
     [Archer.GPUExecFunc, exec_stream]:
       expert i GEMM은 cudaStreamWaitEvent(fetch_event_i) 후 실행
     → 첫 expert가 compute되는 동안 나머지 H2D가 계속 → ★fetch↔compute overlap★
```

**경우 B: cap < demand (overflow, chunk1_staging on, 기본값)**
```
t6  [Python] staging_fetch(E_0)   [h2d] host→staging (PCIe)
    for i in 0..N-1:
      [Python] promote_staging(E_i):
            [h2d] cudaStreamWaitEvent(slot_last_compute_event[slot]) ← 직전 점유 GEMM 대기
                  CudaMemcpyAsync D2D  staging→slot  (intra-GPU, ~5µs)
                  record fetch_event
      [Python] staging_fetch(E_{i+1})  [h2d] host→staging  ★다음 것 prefetch★
      [Python] _archer_batch_compute([E_i 슬라이스])  ← blocking (GEMM)
            이게 도는 동안 위 staging_fetch(E_{i+1})의 PCIe가 h2d_stream에서 overlap
```
> 비싼 PCIe(host→staging)는 GEMM과 겹치고, 싼 D2D(staging→slot)만 slot 재사용을 기다림 → cap=1에서도 overlap 유지.
> 주의: 여기서 per-expert `torch.unique().cpu()` sync를 없애는 micro-opt는 cap=1 고빈도에서 archer OutputFunc
> 텐서크기 레이스를 유발해 되돌렸다 — 이 sync는 set_inputs↔OutputFunc 직렬화에 load-bearing.

### ▶ Phase 3c/3h — 결과 a2a 역방향 + scatter (NCCL)
```
t8  [Python] return_outputs(hit_local_output, recv/send_counts):
       [NCCL all_to_all] 결과를 원래 토큰 소유 rank로 되돌림
       → scatter_combine_into(final, ...)  (token_idx 자리에 누적)
t9  [Python] return_outputs(miss_local_output, ...):
       [NCCL all_to_all] → scatter_combine_into(final, ...)
    (hit/miss 결과가 같은 토큰에서 자연 합산)
```

### ▶ Phase 4 — drift 검증 (가드)
```
t10 [Controller] verify_layer_end(layer_id):
       각 rank: shadow(cache_view) == archer.get_cached_experts() ?
       다르면 즉시 RuntimeError raise  ← drift는 correctness 버그라 hard-fail
```

---

## 3. 스트림 오버랩 타임라인 (개념도, cap≥demand)

```
시간 →

NCCL  : [all_gather]      [hit a2a]        [miss a2a]              [hit back][miss back]
            (P1)            (3a)              (3e)                    (3c)     (3h)
                              \                  \
CPU   :                    [plan_misses(P2)]   [launch_fetch_ops]
                              겹침↑

h2d   :                                        [H2D e0][H2D e1][H2D e2]...  ← PCIe, exec와 겹침
                                                   |       |
exec  :              [hit GEMM (3b)]             [GEMM e0][GEMM e1]...        ← fetch_event로 gate
                         ↑ slot_last_compute_event 기록 → 뒤 fetch가 대기
```

핵심 overlap 3쌍:
1. **hit a2a(NCCL) ∥ plan_misses(CPU)** — 통신 중 계획.
2. **PCIe fetch(h2d) ∥ GEMM(exec)** — fetch_event로 순서만 보장, 병렬 진행.
3. **(staging일 때) E_{i+1} PCIe ∥ E_i GEMM** — staging buffer 덕분.

> 단, 측정상 이 EP 구성은 **NCCL a2a가 prefill의 ~68% 지배** → fetch overlap을 잘 해도 total은 a2a에 묶임.
> decode는 a2a가 작아 fast-a2a로 −35%. (자세히는 메모리 `project_ep_a2a_bound`.)

---

## 4. 3주체의 "view" 정리

| | 무엇을 보나 | 언제 갱신 | 진실성 |
|---|---|---|---|
| **Controller (shadow)** | per-rank slot 점유 [(layer,expert)] | Phase 2 plan에서 낙관적 갱신 | 단일 진실 (모든 rank 동일) |
| **Archer (cached_experts_)** | 실제 GPU에 올라온 expert + slot 포인터 | replace_async/promote에서 동기 갱신 | 물리 상태 |
| **Phase 4** | 둘이 일치하는가 | 매 layer 끝 | 불일치=drift=raise |

shadow가 **먼저** 결정하고 archer가 **그대로** 실행 → Phase 4에서 byte-identical. (검증서 전부 drift=0.)

---

## 5. 핵심 동기화 이벤트 (누가 누구를 기다리나)

```
fetch_event              : [h2d 가 record] → [exec 가 wait]   "fetch 끝나야 GEMM이 weight 읽음"
slot_last_compute_event  : [exec 가 record] → [h2d 가 wait]   "GEMM 끝나야 그 slot 덮어씀"(재사용 안전)
NCCL all_to_all          : 동기 collective — 가장 무거운 rank를 모두 기다림 (prefill 병목)
wait_expert()            : Python(CPU) 블로킹 — GEMM+OutputFunc 완료까지
```

---

## 한 줄 요약

매 layer × 매 토큰(decode)마다:
**demand 수집(NCCL all_gather) → hit 토큰 a2a(NCCL) ∥ miss 계획(CPU) → hit GEMM → miss 토큰 a2a(NCCL)
→ miss fetch(PCIe h2d) ∥ compute(GEMM exec) → 결과 a2a(NCCL) → drift 검증**
의 순환. Controller(shadow)가 결정하고 Archer가 slot pool 위에서 실행하며, 세 스트림이 cudaEvent로
겹쳐 돈다.
