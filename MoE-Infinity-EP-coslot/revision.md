# Controller-Owned Slot Execution 최종 TODO + 구현 순서

## 0. 최종 목표

현재 per-expert staging path를 새 구조로 교체한다.

핵심 구조:

* Controller가 모든 miss expert의 `fetcher_rank`, `dst_slot`, `victim`, `order`를 결정한다.
* Archer는 Controller plan을 `PlanQueue`로 받고, slot/victim을 다시 고르지 않는다.
* Archer는 `active_fetch`를 최대 1개만 유지한다.
* Fetch가 끝나 expert가 `ExecQueue`로 넘어간 뒤에야 다음 `PlanQueue` op를 pop한다.
* Target slot이 safe하면 direct H2D.
* Target slot이 busy하면 staging slot으로 H2D 후, slot safe 시 D2D로 commit.
* GEMM 후 `OutputFunc`는 바로 scatter하지 않고 partial output만 저장한다.
* 모든 local expert 실행 완료 후 `combine_partials()`를 수행하고, 이후 EP return all-to-all을 한 번만 한다.
* 기존 `GPUFetchFunc`, `input_queue`, autonomous eviction/fetch fallback은 제거한다.

---

# Phase 0. 복제본 생성 및 작업 격리

## TODO

원본 repo는 건드리지 않는다. 새 복제본에서 작업한다.

```bash
MoE-Infinity-EP        -> MoE-Infinity-EP-coslot
MoE-Infinity-EP-archer -> MoE-Infinity-EP-archer-coslot
```

해야 할 일:

* launcher path를 `-coslot` 기준으로 수정
* `PYTHONPATH`를 `MoE-Infinity-EP-coslot`로 수정
* Archer build path를 `MoE-Infinity-EP-archer-coslot`로 수정
* 원본 repo를 참조하는 import/build artifact가 남아 있으면 fail-fast

## 완료 조건

```bash
python -c "import moe_infinity_ep; print(moe_infinity_ep.__file__)"
```

출력 경로가 `MoE-Infinity-EP-coslot`를 가리켜야 한다.

Archer `_store.so` 또는 `_engine.so`도 coslot build artifact를 참조해야 한다.

---

# Phase 1. Controller planner 구현

## 목표

모델/Archer 없이 Controller가 의도한 FetchPlan을 정확히 만드는지 먼저 검증한다.

## 관련 파일

```text
controller/global_controller.py
controller/rank_planner.py
controller/owner_policy.py
controller/evict_policy.py
controller/layer_plan.py
controller/slot_cache.py
```

---

## 1.1 HitOp / FetchOp 구조 명확화

```python
@dataclass
class HitOp:
    layer: int
    expert: int
    serving_rank: int
    slot: int
    order: int | None = None


@dataclass
class FetchOp:
    layer: int
    expert: int
    fetcher_rank: int
    dst_slot: int
    victim_layer: int | None
    victim_expert: int | None
    order: int
```

---

## 1.2 Phase 1 classify

Controller는 각 layer에서 다음을 수행한다.

```text
1. DemandCollector로 per_rank_count[rank, expert] 수집
2. 모든 rank가 동일한 demand matrix를 갖도록 함
3. demanded expert union 생성
4. cache_view.locate(layer, expert)로 HIT/MISS 분류
```

분류 기준:

```text
HIT:
  이미 어떤 rank의 resident slot에 있음
  HitOp 생성

MISS:
  어느 rank에도 resident하지 않음
  owner_policy로 fetcher_rank 결정
```

owner policy:

```text
naive:
  fetcher_rank = expert_id % ep_size

balanced:
  load-aware argmin
```

---

## 1.3 Phase 2 plan_misses

Rank별 deterministic sequential planner를 사용한다.

```python
ordered = sorted(
    miss_experts,
    key=lambda e: (-demand[e], e.layer, e.expert),
)

order = 0

for expert in ordered:
    if rank_cache.is_resident(expert):
        debug_counter.already_resident_skip += 1
        continue

    empty = rank_cache.first_empty_slot()

    if empty is not None:
        dst_slot = empty
        victim = None
    else:
        victim_slot = evict_policy.pick_victim(rank_cache, ...)
        dst_slot = victim_slot
        victim = rank_cache.slots[victim_slot]

    fetch_ops.append(FetchOp(
        layer=expert.layer,
        expert=expert.expert,
        fetcher_rank=r,
        dst_slot=dst_slot,
        victim_layer=victim.layer if victim else None,
        victim_expert=victim.expert if victim else None,
        order=order,
    ))
    order += 1

    rank_cache.apply(
        slot=dst_slot,
        evict=victim,
        insert=expert,
    )
```

---

## 1.4 Planner invariants

반드시 구현/검증한다.

```text
1. first_empty_slot()은 deterministic해야 한다.
   예: 가장 작은 slot id.

2. evict_policy.pick_victim()도 deterministic해야 한다.
   LRU tie-break 예:
     (last_used, slot_id, layer, expert)

3. rank_cache.apply()는 inserted expert의 last_used를 최신 plan sequence로 갱신한다.

4. FetchOp.order는 rank-local FIFO order다.

5. 모든 rank가 전체 global plan을 동일하게 계산한다.

6. Archer에는 fetcher_rank == my_rank인 op만 submit한다.

7. Hit/Miss는 disjoint해야 한다.
   같은 expert가 hit이면서 miss이면 FATAL.
```

---

## 1.5 Golden Test A — Controller planner golden

### 테스트 입력

```text
cap = 4

initial cache:
  slot0 H0
  slot1 H1
  slot2 O0
  slot3 O1

hit:
  H0, H1

miss order by demand:
  M3, M0, M6, M1, M2, M7
```

### 기대 FetchPlan

```text
op0: M3 -> slot2, victim O0
op1: M0 -> slot3, victim O1
op2: M6 -> slot0, victim H0
op3: M1 -> slot1, victim H1
op4: M2 -> slot2, victim M3
op5: M7 -> slot3, victim M0
```

### 통과 조건

```text
1. dst_slot, victim, order가 golden과 일치
2. 각 FetchOp 생성 직후 cache_view state가 golden과 일치
3. 최종 cache_view가 expected final view와 일치
4. 같은 victim 중복 없음
5. 방금 insert한 expert가 즉시 victim으로 다시 뽑히지 않음
6. 모든 rank의 shadow hash가 동일
```

---

# Phase 2. Archer 자료구조 추가

## 목표

실제 CUDA/GEMM 구현 전에 Archer 내부 구조를 먼저 만든다.

## 관련 파일

```text
core/parallel/expert_dispatcher.h
core/parallel/expert_dispatcher.cpp
```

---

## 2.1 새 자료구조

```cpp
struct PlanOp {
    int layer;
    int expert;
    int dst_slot;
    int victim_layer;
    int victim_expert;
    int order;
};

struct HitOp {
    int layer;
    int expert;
    int slot;
};

struct ExecTask {
    int layer;
    int expert;
    int slot;
    cudaEvent_t fetch_event;
    torch::Tensor token_idx_snapshot;
    torch::Tensor weights_snapshot;
};

struct Partial {
    torch::Tensor output;
    torch::Tensor token_idx;
    torch::Tensor weights;
    int layer;
    int expert;
};
```

---

## 2.2 새 멤버

```cpp
std::deque<PlanOp> plan_queue_[MAX_GPU];
bool has_active_fetch_[MAX_GPU];

std::vector<Partial> partial_outputs_[MAX_GPU];

enum SlotState {
    FREE,
    COMPUTING,
};

std::vector<SlotState> slot_state_[MAX_GPU];

int expected_order_[MAX_GPU];

Queue<ExecTask> exec_queue_[MAX_GPU];

std::atomic<int> pending_;
std::condition_variable pending_cv_;
std::mutex pending_mutex_;
```

---

## 2.3 Staging slot

`InitSlotPool(cap, expert_bytes)`는 `(cap + 1)`개 slot을 할당한다.

```text
slot 0 .. cap-1:
  resident cache slot

slot cap:
  staging reserved slot
```

중요 invariant:

```text
1. staging slot은 cache_view에 절대 들어가지 않는다.
2. staging slot은 key_to_slot / slot_to_key에 등록하지 않는다.
3. staging slot은 Archer 내부 H2D 임시 destination이다.
4. 런타임 cudaMalloc/free는 금지한다.
```

기존 별도 `staging_ptr_` cudaMalloc은 제거한다.
staging도 slot pool의 reserved slot을 재사용한다.

---

# Phase 3. Archer fake scheduler 구현

## 목표

실제 CUDA 복사/GEMM 없이 PlanQueue, active_fetch, ExecQueue 동작만 검증한다.

## 핵심 규칙

```text
1. active_fetch는 최대 1개.
2. active_fetch가 ExecQueue로 넘어가기 전까지 다음 PlanQueue op를 pop하지 않는다.
3. PlanQueue FIFO order는 Controller order와 동일하다.
4. Archer는 dst_slot/victim을 다시 고르지 않는다.
5. staging은 Archer 내부 임시 H2D destination일 뿐 cache_view에 들어가지 않는다.
```

---

## 3.1 start_next_fetch

```cpp
void start_next_fetch(int gpu) {
    if (has_active_fetch_[gpu]) return;
    if (plan_queue_[gpu].empty()) return;

    PlanOp op = plan_queue_[gpu].front();
    plan_queue_[gpu].pop_front();

    FATAL_IF(op.order != expected_order_[gpu]++);

    has_active_fetch_[gpu] = true;

    if (slot_state_[gpu][op.dst_slot] == FREE) {
        launch_direct_fetch(op);
    } else {
        launch_staging_fetch(op);
    }
}
```

---

## 3.2 Fake direct fetch

```cpp
launch_direct_fetch(op):
    // fake H2D done immediately
    commit(op)
    ExecQueue.push(op)
    has_active_fetch = false
    start_next_fetch()
```

---

## 3.3 Fake staging fetch

```cpp
launch_staging_fetch(op):
    // fake host -> staging done

    if slot_state[op.dst_slot] == FREE:
        commit(op)
        ExecQueue.push(op)
        has_active_fetch = false
        start_next_fetch()
    else:
        keep active_fetch waiting
```

---

## 3.4 Fake compute done

```cpp
on_fake_compute_done(task):
    slot_state[task.slot] = FREE

    if active_fetch is waiting for this slot:
        promote/commit active_fetch
        ExecQueue.push(active_fetch)
        has_active_fetch = false
        start_next_fetch()
```

---

## 3.5 Golden Test B — Archer scheduler golden

검증 항목:

```text
1. active_fetch <= 1 항상
2. PlanQueue pop order가 order와 일치
3. direct path와 staging path 모두 발생
4. staging op가 ExecQueue로 넘어가기 전 다음 op pop 없음
5. victim mismatch면 FATAL
6. slot overwrite 없음
7. final slot_to_key가 Controller final cache_view와 일치
```

---

# Phase 4. 실제 Archer CUDA fetch 구현

## 목표

fake scheduler를 실제 h2d_stream 기반으로 바꾼다.

---

## 4.1 direct fetch

```cpp
launch_direct_fetch(op):
    cudaStreamWaitEvent(
        h2d_stream,
        slot_last_compute_event[op.dst_slot]
    );

    cudaMemcpyAsync(
        dst_slot_ptr,
        host_expert_ptr,
        bytes,
        cudaMemcpyHostToDevice,
        h2d_stream
    );

    cudaEventRecord(
        fetch_event[op.expert],
        h2d_stream
    );
```

권장 commit 원칙:

```text
H2D 완료 후 commit.
```

현실적 대안:

```text
metadata early commit + ExecTask fetch_event wait.
```

early commit을 쓰는 경우 반드시 로그에 남긴다.

```text
[WARN] early metadata commit enabled; compute is gated by fetch_event
```

---

## 4.2 staging fetch

```cpp
launch_staging_fetch(op):
    staging_slot = cap;

    cudaMemcpyAsync(
        staging_ptr,
        host_expert_ptr,
        bytes,
        cudaMemcpyHostToDevice,
        h2d_stream
    );

    cudaStreamWaitEvent(
        h2d_stream,
        slot_last_compute_event[op.dst_slot]
    );

    cudaMemcpyAsync(
        dst_slot_ptr,
        staging_ptr,
        bytes,
        cudaMemcpyDeviceToDevice,
        h2d_stream
    );

    cudaEventRecord(
        fetch_event[op.expert],
        h2d_stream
    );

    commit(op);
    ExecQueue.push(op);
    has_active_fetch = false;
    start_next_fetch();
```

중요:

```text
1. direct path도 slot_last_compute_event를 기다린다.
2. staging path는 host->staging H2D 후, dst_slot event를 기다리고 D2D commit한다.
3. slot overwrite safety는 CUDA event가 보장한다.
```

---

## 4.3 commit(op)

```cpp
commit(op):
    FATAL_IF(slot_to_key[op.dst_slot] != key(op.victim))
        log:
          gpu
          dst_slot
          controller_victim
          archer_actual
          expert
          order

    erase victim from key_to_slot
    erase victim from cached_experts

    slot_to_key[op.dst_slot] = key(op.expert)
    key_to_slot[key(op.expert)] = op.dst_slot
    cached_experts.insert(key(op.expert))
```

절대 금지:

```text
1. 다른 slot 찾기
2. 다른 victim 고르기
3. GPUFetchFunc fallback
4. input_queue miss retry
5. FindExpertEvict
```

---

# Phase 5. GPUExecFunc 재작성

## 목표

ExecQueue task를 GEMM하고 partial output만 저장한다.

---

## 5.1 GPUExecFunc worker 수 고정

controller-owned mode에서는 GPU당 exec worker 1개로 강제한다.

```cpp
if (controller_owned_mode) {
    num_exec_workers_per_gpu = 1;
}
```

시작 로그:

```text
[CO_SLOT] exec_workers_per_gpu=1
```

FATAL 조건:

```text
if exec_workers_per_gpu != 1:
    FATAL
```

---

## 5.2 Exec flow

```cpp
GPUExecFunc:
    task = ExecQueue.pop()

    FATAL_IF(slot_to_key[task.slot] != key(task.expert))

    slot_state[task.slot] = COMPUTING

    cudaStreamWaitEvent(
        exec_stream,
        task.fetch_event
    );

    output = MoEMLP.forward(
        input_slice,
        slot_weight_ptr,
        exec_stream
    );

    OutputFunc_store_partial(task, output)

    cudaEventRecord(
        slot_last_compute_event[task.slot],
        exec_stream
    );

    slot_state[task.slot] = FREE

    pending_--

    if pending_ == 0:
        pending_cv_.notify_all()
```

---

## 5.3 pending decrement 위치

`pending_--`는 반드시 아래 이후에 수행한다.

```text
1. GEMM 완료
2. partial output 저장 완료
3. slot_last_compute_event record
4. slot_state FREE
```

---

# Phase 6. OutputFunc_store_partial 구현

## 목표

기존 즉시 scatter OutputFunc를 partial 저장 구조로 바꾼다.

---

## 6.1 기존 OutputFunc 제거/대체

기존:

```text
output * router_weight
final_hidden_states.index_add_
pending--
```

새 구조:

```text
Partial 저장만 수행
pending--는 GPUExecFunc에서 수행
```

---

## 6.2 Partial에 저장할 것

```cpp
Partial {
    torch::Tensor output;       // [K_e, H]
    torch::Tensor token_idx;    // [K_e]
    torch::Tensor weights;      // [K_e] or [K_e, 1]
    int layer;
    int expert;
};
```

---

## 6.3 Snapshot rule

`OutputFunc_store_partial`는 shared `router_mask_`를 늦게 다시 보면 안 된다.

따라서 `ExecTask` 생성 시점에 다음을 snapshot으로 만든다.

```text
token_idx_snapshot
weights_snapshot
token_count
```

주의:

```text
token_count == 0이면 ExecTask 생성 금지.
```

---

# Phase 7. WaitLayerDone + combine_partials

## 7.1 WaitLayerDone

```cpp
WaitLayerDone():
    pending_cv_.wait(pending_ == 0)

    final_local = combine_partials()

    return final_local
```

---

## 7.2 combine_partials

```cpp
final_hidden_states.zero_()

sort partial_outputs by deterministic order:
    (layer, expert, insertion_order)

for partial in partial_outputs:
    weighted = partial.output * partial.weights
    final_hidden_states.index_add_(
        0,
        partial.token_idx,
        weighted
    )

final_hidden_states = final_hidden_states.to(model_dtype)

partial_outputs.clear()

return final_hidden_states
```

---

## 7.3 dtype invariant

```text
combine accumulation dtype = fp32 권장
return dtype = model dtype, e.g. bf16
```

FATAL 조건:

```text
final_local dtype != model dtype before a2a
```

---

## 7.4 stream sync

`wait_layer_done()`가 Python에 반환하기 전에 `final_local`이 완성되어 있어야 한다.

초기 구현은 안전하게 아래 중 하나를 사용한다.

```cpp
cudaStreamSynchronize(exec_stream)
```

또는 equivalent CUDA event wait.

추후 최적화 가능.

---

# Phase 8. Python binding 추가

## 파일

```text
core/python/py_archer_prefetch.cpp
```

---

## 추가 API

```python
init_slot_pool(cap, expert_bytes)
set_inputs(recv_hidden, recv_router_mask, recv_weights, is_decode)
submit_plan(gpu_id, hit_ops, miss_ops)
wait_layer_done()
get_cached_experts()
```

---

## 제거 또는 deprecated

```python
enqueue_expert
set_expected_queue
explicit_fetch
explicit_replace_async
fetch_to_staging
promote_staging
```

완전 제거 전에는 controller-owned mode에서 호출 시 FATAL.

```text
controller-owned mode에서는 호출 금지
```

---

# Phase 9. Python CommandDispatcher 재작성

## 파일

```text
controller/command_dispatcher.py
```

---

## 새 API

```python
submit_plan(plan, hit_ops, my_rank)
```

---

## 해야 할 일

```python
local_hit_ops = [
    hit for hit in hit_ops
    if hit.serving_rank == my_rank
    and received_token_count(hit.expert) > 0
]

local_miss_ops = [
    op for op in plan.fetch_ops
    if op.fetcher_rank == my_rank
    and received_token_count(op.expert) > 0
]
```

주의:

```text
token_count == 0 expert는 SubmitPlan에 넣지 않는다.
```

Archer binding으로 변환한다.

```python
archer.submit_plan(
    0,
    local_hit_ops,
    local_miss_ops,
)
```

---

# Phase 10. ep_executor `_coop_dispatch` 재작성

## 제거 대상

```python
_archer_batch_compute
_pipelined_miss_compute
_staging_pipeline_miss_compute
```

---

## 새 흐름

```python
p1 = controller.classify_and_kick_hit(...)
plan = controller.plan_misses(...)

expert_to_rank = merge_hit_owner_and_miss_fetcher(
    p1,
    plan,
)

table = build_expert_rank_table(
    expert_to_rank,
)

recv = route_tokens(
    pack_tokens(
        hidden,
        full_router_mask,
        table,
        ...
    )
)

local_dispatcher.set_inputs(
    recv.hidden,
    recv.router_mask,
    recv.weights,
    is_decode,
)

command_dispatcher.submit_plan(
    plan,
    p1.hit_ops,
    my_rank,
)

final_local = local_dispatcher.wait_layer_done()

send_back = return_outputs(
    final_local,
    recv_counts,
    send_counts,
    ...
)

scatter_combine_into(
    final,
    send_back,
    ...
)

controller.verify_layer_end()
```

---

## collective symmetry

모든 rank는 반드시 다음을 호출한다.

```text
route_tokens 1회
return_outputs 1회
```

local plan이 비어 있어도 collective는 호출한다.

---

# Phase 11. nvlink_router 최소 변경

## 유지

```python
route_tokens
return_outputs
scatter_combine_into
derive_split_counts
```

---

## 변경

hit/miss를 분리하지 않고 통합 expert_to_rank table을 사용한다.

```text
hit expert -> resident owner rank
miss expert -> fetcher rank
```

검증:

```python
assert no_overlap(hit_experts, miss_experts)
assert every_demanded_expert_has_dst_rank
assert (dst_ranks >= 0).all()
```

---

# Phase 12. Golden tests 구현

## Test A — Controller planner golden

Python only.

검증:

```text
1. 매 FetchOp 생성 후 cache_view state
2. dst_slot
3. victim
4. order
5. final cache_view
```

---

## Test B — Archer scheduler golden

C++ fake event/no GEMM.

검증:

```text
1. active_fetch <= 1
2. PlanQueue FIFO
3. direct path
4. staging path
5. commit
6. ExecQueue push
7. victim mismatch FATAL
```

---

## Test C — Python↔Archer binding golden

dummy tensors.

절차:

```text
1. init_slot_pool(cap=4)
2. initial slot state 설정
3. submit_plan(hit, miss)
4. wait_layer_done()
5. get_cached_experts()
```

검증:

```text
Archer final slots == Controller final cache_view
pending == 0
partial_outputs cleared
```

---

## Test D — single-rank MoE correctness

작은 hidden dim과 dummy expert output 사용.

Reference:

```python
final[token] = sum(output_e[token] * weight_e)
```

검증:

```text
final_local == reference
```

---

## Test E — 2-rank a2a symmetry smoke

모델 없이 route/return만.

검증:

```text
1. 모든 rank route 1회
2. 모든 rank return 1회
3. empty local plan rank도 collective 참여
4. NCCL hang 없음
```

---

# Phase 13. 실제 모델 검증

Golden A–E 전부 통과 후에만 실제 모델 실행.

---

## 13.1 30B 회귀

```text
qwen3_30b
cap large
1-step e2e
drift=0
decode 정상
```

---

## 13.2 235B cap8

```text
235B
cap=8
naive + balanced
prefill64
decode32
bs16
n_measure48
NUMA shared ON
```

성공 기준:

```text
1. decode 완주
2. progress 출력
3. drift=0
4. FATAL 없음
5. hang 없음
6. NaN/Inf 없음
```

---

# Phase 14. 진단 로그

env-gate로 다음 로그 유지.

```text
[SubmitPlan] gpu num_hit num_miss pending
[FetchStart] order expert dst_slot victim mode=direct|staging
[FetchCommit] expert dst_slot
[ExecPop] expert slot token_count
[GemmDone] expert slot
[PartialStored] expert token_count
[LayerDone] pending=0 num_partials final_shape dtype
```

---

# Phase 15. Fail-fast 조건

즉시 중단해야 하는 조건:

```text
1. commit 시 slot_to_key[dst_slot] != victim
2. ExecTask 실행 시 slot_to_key[slot] != expert
3. HitOp submit 시 slot_to_key[slot] != expert
4. PlanQueue order 불일치
5. active_fetch overwrite
6. pending_ underflow
7. partial output shape mismatch
8. token_count == 0 task 생성
9. final_local dtype != model dtype before a2a
10. layer end cache_view != archer.get_cached_experts()
```

---

# 구현 순서 요약

```text
Phase 0  복제본 생성
Phase 1  Controller planner + Test A
Phase 2  Archer 자료구조 추가
Phase 3  Archer fake scheduler + Test B
Phase 4  실제 CUDA fetch/direct/staging 구현
Phase 5  GPUExecFunc 1-worker + ExecTask 실행
Phase 6  OutputFunc_store_partial
Phase 7  WaitLayerDone + combine_partials
Phase 8  Python binding
Phase 9  CommandDispatcher submit_plan
Phase 10 ep_executor 통합 a2a + wait_layer_done
Phase 11 nvlink_router 최소 수정
Phase 12 Golden Test A–E
Phase 13 실제 모델 검증
Phase 14 진단 로그 정리
Phase 15 fail-fast 조건 고정
```

---

# 최종 핵심 invariant

```text
1. Controller가 dst_slot/victim/order를 결정한다.
2. Archer는 slot/victim을 다시 고르지 않는다.
3. active_fetch는 최대 1개다.
4. staging은 Archer 내부 임시 H2D destination일 뿐이다.
5. ExecQueue task는 반드시 slot_to_key[slot] == expert여야 한다.
6. slot overwrite safety는 slot_last_compute_event가 보장한다.
7. OutputFunc는 partial 저장만 한다.
8. 모든 expert 완료 후 layer 끝에 combine한다.
9. 모든 rank는 route_tokens 1회, return_outputs 1회를 대칭 호출한다.
10. layer 끝에는 Controller cache_view == Archer get_cached_experts여야 한다.
```
