# Controller-Owned Slot Execution 보강 계획

## 0. 현재 plan에서 반드시 보강해야 할 핵심

기존 plan의 큰 방향은 유지한다.

```text
Controller:
  global demand를 보고 hit/miss를 나누고,
  모든 miss expert의 fetcher_rank, dst_slot, victim, order를 정한다.

Archer:
  Controller가 준 hit_ops와 miss_ops를 SubmitPlan으로 받아 실행한다.
  slot/victim을 다시 고르지 않는다.
  GPUFetchFunc/autonomous fetch fallback을 제거한다.
  expert output은 partial로 저장하고 layer 끝에 한 번 combine한다.
```

다만 다음 부분을 더 명확히 해야 한다.

```text
1. SubmitPlan에서 hit_ops와 miss_ops가 어떻게 구분되어 들어가는가?
2. hit/miss 통합 a2a 후, Archer가 어떤 token slice를 어떤 expert task로 실행하는가?
3. OutputFunc_store_partial이 저장해야 하는 metadata는 무엇인가?
4. final_local combine 후 return_outputs a2a가 기존 owner rank로 정확히 돌아가는가?
5. GPUExecFunc worker가 정말 GPU당 1개인지 검증/강제하는가?
6. 실제 model forward 전에 Python view ↔ Archer view의 slot-plan 기작이 맞는지 어떻게 테스트하는가?
```

---

# 1. SubmitPlan의 hit/miss 구분

## 1.1 Controller 단계에서의 hit/miss

Controller는 Phase 1에서 global demand를 모은다.

```text
per_rank_count[rank, expert]
```

그리고 각 `(layer, expert)`를 `cache_view.locate()`로 조회한다.

```text
if expert is resident in some rank:
    HIT
else:
    MISS
```

여기서 HIT/MISS는 "이번 layer에서 어떤 expert가 이미 GPU resident인가"의 의미다.

---

## 1.2 hit expert의 owner rank

HIT expert는 이미 어떤 rank의 어떤 slot에 있다.

예:

```text
H0 is resident at rank0 slot0
H1 is resident at rank0 slot1
H2 is resident at rank2 slot5
```

따라서 hit expert의 실행 rank는 이미 정해져 있다.

```text
hit_owner_rank[H0] = rank0
hit_owner_slot[H0] = slot0
```

Controller는 hit에 대해 Archer fetch plan을 만들지 않는다. 대신 hit execution op를 만든다.

```text
HitOp:
  expert = H0
  serving_rank = rank0
  slot = slot0
```

---

## 1.3 miss expert의 fetcher rank와 slot

MISS expert는 resident cache에 없다. Controller의 owner policy가 fetcher rank를 결정한다.

```text
naive:
  fetcher_rank = expert_id % ep_size

balanced:
  fetcher_rank = load-aware selected rank
```

그 다음 `RankPlanner`가 해당 rank 안에서 slot/victim을 결정한다.

```text
FetchOp:
  expert = M3
  fetcher_rank = rank0
  dst_slot = slot2
  victim = O0
  order = 0
```

---

## 1.4 SubmitPlan에 들어가는 local hit_ops / miss_ops

각 rank는 전체 global plan을 알고 있지만, Archer에 넘기는 것은 **자기 rank가 실행할 op만**이다.

예를 들어 현재 process가 rank0이면:

```text
local_hit_ops =
  all HitOp where serving_rank == rank0 and this rank received tokens for that expert

local_miss_ops =
  all FetchOp where fetcher_rank == rank0 and this rank received tokens for that expert
```

중요하다.

```text
HitOp/MissOp는 expert가 이 rank에 token을 실제로 받았을 때만 SubmitPlan에 포함한다.
token이 0개인 expert는 실행하지 않는다.
```

즉 SubmitPlan은 다음 형태다.

```python
archer.submit_plan(
    gpu_id=0,
    hit_ops=local_hit_ops,
    miss_ops=local_miss_ops,
)
```

---

# 2. hit/miss 통합 a2a와 token ownership

## 2.1 기존 구조의 문제

이전 구조는 hit와 miss를 분리해 a2a를 두 번 수행했다.

```text
hit route_tokens
miss route_tokens
hit return_outputs
miss return_outputs
```

새 구조에서는 hit/miss를 통합한다.

```text
full route_tokens 1회
full return_outputs 1회
```

---

## 2.2 expert_to_rank 구성

통합 routing을 위해 각 demanded expert의 serving rank를 하나로 정한다.

```text
if expert is HIT:
    serving_rank = hit_owner_rank

if expert is MISS:
    serving_rank = miss_fetcher_rank
```

따라서:

```text
expert_to_rank = hit_owner ∪ miss_fetcher
```

충돌이 없어야 한다.

```text
한 expert가 HIT이면 MISS가 아니다.
한 expert가 MISS이면 HIT가 아니다.
```

검증:

```python
assert no duplicated expert in hit and miss
assert every demanded expert has expert_to_rank
assert (dst_ranks >= 0).all()
```

---

## 2.3 통합 a2a 후 Archer input

통합 `route_tokens` 후 각 rank는 다음을 받는다.

```text
recv_hidden
recv_router_mask
recv_weights
recv_token_owner_info / return routing metadata
```

이 rank가 실행해야 할 expert들의 token들이 `recv_hidden` 안에 모여 있다.

Archer의 `set_inputs`는 이 rank-local routed token batch를 저장한다.

```python
local_dispatcher.set_inputs(
    recv_hidden,
    recv_router_mask,
    recv_weights,
    is_decode,
)
```

이후 SubmitPlan의 각 HitOp/MissOp는 `recv_router_mask`를 통해 자신이 처리할 token slice를 찾는다.

---

# 3. Archer SubmitPlan 세부 동작

## 3.1 SubmitPlan 입력

```cpp
SubmitPlan(gpu_id, hit_ops, miss_ops)
```

각 op는 최소 다음 정보를 갖는다.

```text
HitOp:
  layer
  expert
  slot

PlanOp / MissOp:
  layer
  expert
  dst_slot
  victim_layer
  victim_expert
```

Archer는 hit_ops와 miss_ops를 구분해서 처리한다.

---

## 3.2 SubmitPlan 초기화

SubmitPlan 호출 시:

```text
pending_ = len(hit_ops) + len(miss_ops)
partial_outputs_.clear()
plan_queue_ = miss_ops
active_fetch = None
```

그리고 hit_ops는 바로 ExecQueue로 보낸다.

```text
for hit in hit_ops:
    assert slot_to_key[hit.slot] == hit.expert
    ExecQueue.push(hit.expert, hit.slot, no_fetch_event_or_already_ready)
```

Miss는 PlanQueue에 들어간다.

```text
plan_queue_ = [miss_op0, miss_op1, ...]
```

그 다음 fetch pipeline을 시작한다.

```text
start_next_fetch()
```

---

# 4. PlanQueue와 active_fetch

## 4.1 핵심 원칙

PlanQueue는 무조건 pop하지 않는다.

```text
active_fetch == None일 때만 PlanQueue에서 다음 op를 pop한다.
```

즉:

```text
active_fetch != None:
    do not pop next op

active_fetch == None:
    pop next op and start fetch
```

---

## 4.2 active_fetch의 의미

active_fetch는 "현재 FetchEngine이 처리 중인 miss op"이다.

direct path에서는:

```text
active_fetch = op
H2D host -> dst_slot
commit
ExecQueue.push(op)
active_fetch = None
start_next_fetch()
```

staging path에서는:

```text
active_fetch = op
H2D host -> staging
wait dst_slot safe
D2D staging -> dst_slot
commit
ExecQueue.push(op)
active_fetch = None
start_next_fetch()
```

따라서 staging이 busy한 상태는 별도 scheduler branch가 아니라:

```text
active_fetch가 아직 끝나지 않은 상태
```

로 본다.

---

# 5. Direct fetch path

## 5.1 언제 direct인가?

Fetch 시작 시점에 dst_slot이 safe하면 direct path를 사용한다.

```text
slot_state[dst_slot] == FREE
```

단, 실제 overwrite 안전은 CPU state가 아니라 CUDA event로 보장한다.

---

## 5.2 direct fetch sequence

```text
direct_fetch_to_slot(op):
    assert slot_to_key[op.dst_slot] == op.victim

    h2d_stream:
        cudaStreamWaitEvent(slot_last_compute_event[op.dst_slot])
        H2D host -> dst_slot
        record fetch_event[op.expert]

    wait/callback until H2D done if using commit-after-complete

    commit:
        remove victim
        slot_to_key[op.dst_slot] = op.expert
        key_to_slot[op.expert] = op.dst_slot
        cached_experts.add(op.expert)

    ExecQueue.push(op.expert, op.dst_slot, fetch_event)

    active_fetch = None
    start_next_fetch()
```

권장: D2D/H2D 완료 후 resident commit.
단, 구현상 early metadata update를 쓰면 반드시 ExecQueue task가 fetch_event를 wait해야 한다.

---

# 6. Staging fetch path

## 6.1 staging을 쓰는 이유

staging은 Controller가 모르는 Archer 내부 임시 buffer다.

목적:

```text
dst_slot이 compute 중이라 H2D destination으로 바로 사용할 수 없을 때,
host->GPU H2D를 staging으로 먼저 진행하기 위함.
```

즉 staging은 cache placement policy가 아니다.

---

## 6.2 staging path sequence

```text
fetch_to_staging_then_promote(op):
    assert active_fetch == op

    h2d_stream:
        H2D host -> staging
        record staging_fetch_event

        cudaStreamWaitEvent(slot_last_compute_event[op.dst_slot])
        D2D staging -> op.dst_slot
        record fetch_event[op.expert]

    wait/callback until D2D done if using commit-after-complete

    commit:
        assert slot_to_key[op.dst_slot] == op.victim
        remove victim
        slot_to_key[op.dst_slot] = op.expert
        key_to_slot[op.expert] = op.dst_slot
        cached_experts.add(op.expert)

    ExecQueue.push(op.expert, op.dst_slot, fetch_event)

    active_fetch = None
    start_next_fetch()
```

중요:

```text
staging에 fetch된 expert는 commit 전까지 resident가 아니다.
active_fetch가 ExecQueue로 넘어가기 전까지 다음 op는 pop하지 않는다.
```

---

# 7. ExecQueue와 GPUExecFunc

## 7.1 ExecTask

ExecTask는 다음 정보를 가져야 한다.

```text
ExecTask:
  expert
  slot
  fetch_event
  token_idx_snapshot
  weight_snapshot
```

token_idx/weight는 OutputFunc_store_partial에서 필요하다.

---

## 7.2 GPUExecFunc

GPUExecFunc는 다음 순서로 실행한다.

```text
GPUExecFunc:
    task = ExecQueue.pop()

    slot_state[task.slot] = COMPUTING

    exec_stream:
        cudaStreamWaitEvent(task.fetch_event)
        GEMM / SwiGLU
        OutputFunc_store_partial(task)

        cudaEventRecord(slot_last_compute_event[task.slot])

    slot_state[task.slot] = FREE

    pending_--

    if pending_ == 0:
        pending_cv.notify_all()
```

주의:

```text
slot_state=FREE는 CPU scheduling hint.
실제 overwrite safety는 slot_last_compute_event가 보장.
```

---

# 8. OutputFunc_store_partial

## 8.1 기존 OutputFunc와의 차이

기존 OutputFunc:

```text
expert output
→ router weight 곱
→ final_hidden_states index_add
→ pending--
```

새 OutputFunc_store_partial:

```text
expert output
→ partial_outputs에 저장
→ pending--
```

즉 즉시 scatter하지 않는다.

---

## 8.2 저장해야 하는 정보

```cpp
Partial {
    torch::Tensor output;      // [K_e, H]
    torch::Tensor token_idx;   // [K_e], recv_hidden 기준 local token index
    torch::Tensor weights;     // [K_e] or [K_e, 1]
    int layer;
    int expert;
}
```

중요:

```text
token_idx와 weights는 expert별 snapshot이어야 한다.
shared router_mask_를 나중에 다시 보면 안 된다.
```

이렇게 해야 per-expert shared input race를 피한다.

---

# 9. WaitLayerDone과 combine_partials

## 9.1 WaitLayerDone

```text
WaitLayerDone():
    wait pending_ == 0
    final_local = combine_partials()
    return final_local
```

pending은 이 rank가 이번 layer에서 실행해야 하는 local expert task 수다.

```text
pending = len(hit_ops) + len(miss_ops)
```

---

## 9.2 combine_partials

combine은 Archer C++에서 수행한다.

```text
combine_partials():
    final_hidden_states.zero_()

    sort partial_outputs by deterministic order:
        (layer, expert, maybe insertion_order)

    for partial in partial_outputs:
        weighted = partial.output * partial.weights
        final_hidden_states.index_add_(0, partial.token_idx, weighted)

    cast final_hidden_states to model dtype

    clear partial_outputs

    return final_hidden_states
```

dtype 주의:

```text
accumulate dtype: float32 권장
return dtype: model dtype, e.g. bf16
```

stream 주의:

```text
combine_partials가 exec_stream에서 수행되면,
return_outputs a2a 전에 final_local 완성 event를 보장해야 한다.
```

---

# 10. Python _coop_dispatch 변경

## 10.1 기존 제거

다음은 제거한다.

```text
_archer_batch_compute
_pipelined_miss_compute
_staging_pipeline_miss_compute
per-expert wait_expert loop
```

## 10.2 새 흐름

```text
1. Controller classify:
   p1 = classify_and_kick_hit(...)

2. Controller plan:
   plan = plan_misses(...)

3. expert_to_rank 구성:
   hit expert -> hit owner rank
   miss expert -> fetcher rank

4. full route_tokens 1회:
   hidden/router_mask/weights를 serving rank로 보냄

5. local_dispatcher.set_inputs(recv_hidden, recv_router_mask, recv_weights)

6. command_dispatcher.submit_plan(plan, hit_ops, my_rank)

7. final_local = wait_layer_done()

8. return_outputs(final_local) 1회

9. scatter_combine_into(final)

10. verify_layer_end()
```

---

# 11. SubmitPlan에서 hit/miss local filtering

Python CommandDispatcher는 다음을 해야 한다.

```text
local_hit_ops =
    hit ops where hit_owner_rank == my_rank
    and received token count for expert > 0

local_miss_ops =
    fetch ops where fetcher_rank == my_rank
    and received token count for expert > 0
```

빈 expert는 실행하지 않는다.

하지만 모든 rank는 a2a collective를 대칭적으로 호출해야 한다.

---

# 12. GPUExecFunc worker 수 검증

현재 코드가 `num_threads`에 따라 GPU당 여러 GPUExecFunc를 만들 수 있으면, 새 구조에서는 위험하다.

위험:

```text
partial_outputs 동시 push
slot_state 동시 write
modules_[gpu_id]->SetTensorsFromIds 동시 호출
```

따라서 controller-owned mode에서는 반드시 GPU당 1 exec worker로 고정한다.

```text
assert num_exec_workers_per_gpu == 1
```

런타임 로그:

```text
[CO_SLOT] exec_workers_per_gpu=1
```

검증 항목:

```text
1. 생성된 GPUExecFunc thread 수
2. 생성된 exec_stream 수
3. 동일 gpu_id에 대해 worker가 1개인지
```

---

# 13. 실제 forward 전 테스트

모델 forward 전에 아래 테스트를 먼저 수행한다.

## Test A. Controller planner unit test

초기 cache:

```text
slot0 H0
slot1 H1
slot2 O0
slot3 O1
```

demand:

```text
hit: H0, H1
miss: M3, M0, M6, M1
```

기대 plan:

```text
M3 -> slot2
M0 -> slot3
M6 -> slot0
M1 -> slot1
```

검증:

```text
FetchOp dst_slot/victim/order
Controller final cache_view
```

---

## Test B. Archer plan scheduler toy test

실제 GEMM 없이 fake sleep/event로 PlanQueue/active_fetch를 테스트한다.

검증:

```text
active_fetch는 항상 최대 1개
PlanQueue pop 순서 유지
direct path와 staging path 모두 동작
staging op가 ExecQueue로 넘어가기 전 다음 op pop 없음
slot overwrite 없음
```

---

## Test C. Python↔Archer binding test

작은 dummy tensors로:

```text
init_slot_pool(cap=4)
set initial slot state
submit_plan(hit_ops, miss_ops)
wait_layer_done()
get_cached_experts()
```

검증:

```text
Archer final slots == Controller final cache_view
pending == 0
partial_outputs cleared
```

---

## Test D. Single-rank fake MoE output correctness

작은 hidden_dim과 dummy expert function으로 reference 계산과 비교한다.

```text
expected[token] =
    sum(output_expert_i[token] * router_weight_i)
```

검증:

```text
final_local == reference
```

---

## Test E. 2-rank a2a symmetry smoke

모델 forward 없이 route/return만 테스트한다.

검증:

```text
all ranks call route_tokens once
all ranks call return_outputs once
empty local plan rank도 collective 참여
no NCCL hang
```

---

# 14. Fail-fast / diagnostics

새 경로에는 다음 로그가 있어야 한다.

```text
[SubmitPlan]
  gpu
  num_hit
  num_miss
  pending

[FetchStart]
  order
  expert
  dst_slot
  victim
  mode=direct|staging

[FetchCommit]
  expert
  dst_slot
  active_fetch cleared

[ExecPop]
  expert
  slot

[GemmDone]
  expert
  slot

[PartialStored]
  expert
  token_count

[LayerDone]
  pending=0
  num_partials
  final_local shape
```

Fail-fast 조건:

```text
1. slot_to_key[dst_slot] != victim at commit
2. ExecTask slot_to_key[slot] != expert
3. pending_ underflow
4. active_fetch overwritten
5. partial output shape mismatch
6. final_local dtype mismatch before a2a
```

---

# 15. Summary

최종 구조는 다음과 같다.

```text
Controller:
  hit/miss classify
  expert_to_rank 결정
  miss FetchPlan 생성
  cache_view 최종 shadow 갱신

Python:
  hit+miss 통합 a2a 1회
  Archer.set_inputs
  Archer.submit_plan
  Archer.wait_layer_done
  return_outputs a2a 1회

Archer:
  hit_ops -> ExecQueue
  miss_ops -> PlanQueue
  active_fetch 하나로 direct/staging fetch
  fetch 완료 후 ExecQueue
  GPUExecFunc가 GEMM
  OutputFunc_store_partial
  pending--
  pending==0이면 combine_partials
  final_local 반환
```

핵심 invariant:

```text
1. Controller가 dst_slot/victim/order를 정한다.
2. Archer는 slot/victim을 다시 고르지 않는다.
3. active_fetch는 최대 1개다.
4. staging은 Archer 내부 임시 H2D destination일 뿐이다.
5. ExecQueue task는 반드시 slot_to_key[slot] == expert여야 한다.
6. slot overwrite safety는 slot_last_compute_event가 보장한다.
7. OutputFunc는 partial 저장만 하고, layer 끝에 combine한다.
8. 모든 rank는 route_tokens 1회, return_outputs 1회를 대칭 호출한다.
```
