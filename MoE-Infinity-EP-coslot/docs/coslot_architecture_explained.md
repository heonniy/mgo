# MoE-Infinity-EP-coslot 동작 기작 완전 해설

> 이 문서는 `MoE-Infinity-EP-coslot` 레포지토리가 **어떻게 코드가 나뉘어 있고, 누가 누구에게 명령을 내리며, 그 명령이 실제로 어떻게 실행되는지**, 그리고 **a2a(all-to-all)가 언제 발생하고, compute stream과 fetch stream이 어떻게 서로를 가리는지(overlap)** 를 처음 보는 사람도 따라올 수 있도록 단계별로 설명합니다.
>
> 핵심 코드 라인은 `파일:라인` 형태로 인용했고, 실측·결정 사항은 본문에 표시했습니다.

---

## 0. 한 문단 요약

이 레포는 **Qwen3 MoE 모델을 여러 GPU에 Expert-Parallel(EP)로 펼쳐서, 전문가(expert) weight를 디스크/호스트RAM에 offload 해두고 필요한 것만 GPU로 가져오며(fetch) 추론**하는 시스템입니다. 핵심 아이디어는 **"컨트롤러가 슬롯 실행을 소유한다(Controller-Owned Slot Execution, coslot)"** 입니다. 즉,

- **Python 쪽(이 레포)** 은 "어떤 전문가가 어느 GPU에 있고, 무엇을 새로 가져와야 하고, 토큰을 어디로 보내야 하는지"를 **계획(plan)** 합니다. → *두뇌*
- **archer C++ 백엔드(별도 레포 `-archer-coslot`)** 는 그 계획을 받아 **실제 weight를 GPU로 복사(fetch)하고 GEMM을 돌리고 결과를 합칩니다.** → *손발*

두 레이어 사이의 유일한 명령 통로는 `submit_plan()` 한 개입니다. Python은 매 레이어마다 "이번 레이어의 전체 계획표"를 한 번에 넘기고, archer가 그 안에서 fetch와 compute를 알아서 스트림으로 겹쳐 돌립니다.

---

## 1. 큰 그림: 코드가 어떻게 나뉘어 있나

레포는 `moe_infinity_ep/` 아래 6개 서브패키지로 깔끔하게 역할 분리되어 있습니다.

```
moe_infinity_ep/
├── launch/        ← 프로세스 띄우기 · 분산 초기화 · NUMA 공유 호스트 메모리
│   ├── entry.py               (MoE_EP: 모델 적재 오케스트레이션)
│   ├── distributed_setup.py   (GPU pin, NCCL init_process_group)
│   ├── numa_shared_host.py    (memfd 공유 + cudaHostRegister)
│   └── init_logging.py        (cold-start 구조적 로깅)
│
├── controller/    ← ★ 두뇌. coslot 계획 수립 (단일 컨트롤러 경로)
│   ├── global_controller.py   (GlobalCacheController: 전체 흐름 지휘)
│   ├── slot_cache.py          (슬롯=물리 자원 모델, shadow cache)
│   ├── demand_collector.py    (Phase1.1: NCCL all_gather 수요 수집)
│   ├── owner_policy.py         (Phase1.4: 어느 rank가 어느 expert를 맡나)
│   ├── rank_planner.py         (Phase2: rank별 fetch/evict 순차 계획)
│   ├── evict_policy.py         (LRU/LFU/Demand-aware victim 선택)
│   ├── layer_plan.py           (HitOp/FetchOp/LayerPlan 자료구조)
│   ├── command_dispatcher.py   (★ archer로 가는 유일한 다리: submit_plan)
│   └── priority_aggregator.py  (선택: EAMC 우선순위, 기본 off)
│
├── runtime/       ← 실행 엔진. 컨트롤러를 호출하고 archer를 구동
│   ├── distributed_engine.py  (DistributedOffloadEngine: 엔진 생명주기·배선)
│   └── ep_executor.py         (★ EPExpertExecutor: 한 MoE 레이어 실행)
│
├── exec/          ← 통신 레이어
│   └── nvlink_router.py       (★ a2a 패킹/라우팅, pack/route/return)
│
├── models/        ← HuggingFace 모듈 교체용 후크
│   ├── qwen_ep.py             (Qwen3MoEBlockEP.forward → executor 호출)
│   └── qwen_decoder_ep.py     (decoder layer shim, 현재 identity)
│
├── instrument/    ← 계측 (a2a/fetch 시간, hit/miss, drift)
│   ├── counters.py
│   └── trace.py
│
└── utils/
    ├── config.py              (YAML 파싱, ep_size==world_size 강제)
    └── expert_specs.py        (expert당 byte, slot cap 설계)
```

**역할을 한 문장씩으로:**

| 레이어 | 한 일 | 비유 |
|--------|-------|------|
| `launch/` | 프로세스를 GPU 수만큼 띄우고, NCCL를 열고, 호스트RAM에 모델을 한 번만 올려 공유 | 무대 설치 |
| `controller/` | "이번 레이어, 누가 뭘 갖고 있고 / 뭘 새로 가져와야 하나" 결정 | 지휘자 |
| `runtime/` | 레이어마다 컨트롤러에게 계획을 받아 통신·archer 실행을 엮음 | 연주자 |
| `exec/` | 토큰을 전문가 주인 rank로 보내고 결과를 되받는 a2a | 우편배달 |
| archer C++ | 실제 weight를 GPU로 복사하고 GEMM 실행 | 주방 |

> **단일 컨트롤러 경로 (중요).** 과거 `pin_manager` / `fetch_policy` / `cache.sync` 같은 레거시 경로는 전부 제거됐습니다. 지금은 `GlobalCacheController` **하나**가 모든 fetch/evict 결정을 내립니다 (`qwen_decoder_ep.py`의 pipeline-overlap도 2026-05-27에 제거되어 현재는 identity shim, [qwen_decoder_ep.py:55](../moe_infinity_ep/models/qwen_decoder_ep.py#L55)).

---

## 2. 시작부터: 프로세스는 어떻게 떠서 준비되나

### 2.1 프로세스 기동 (torchrun + numa_wrap)

진입은 `scripts/m*.py` (예: `m3_forward.py`)이고, 실행 스크립트(`run_m3.sh`)가 **torchrun**으로 GPU 수만큼 프로세스를 띄웁니다 (mp.spawn 아님):

```bash
setsid torchrun --standalone --nproc_per_node=$NPROC --no-python \
    scripts/numa_wrap.sh scripts/m3_forward.py "$CONFIG"
```

- `torchrun`이 각 자식에게 `RANK / WORLD_SIZE / LOCAL_RANK` 환경변수를 주입합니다.
- `numa_wrap.sh`가 각 rank를 **`numactl --cpunodebind=N --membind=N`** 으로 감싸서, 파이썬이 뜨기 전에 GPU의 NUMA 노드에 CPU·메모리를 고정합니다.
- `setsid`로 세션을 분리합니다 (archer C++의 atexit hang 회피, [AIO lost-wakeup 메모 참조]).

### 2.2 CUDA import 전에 GPU를 1개로 묶기 (순서가 생명)

`pin_visible_device()`는 **`moe_infinity` import보다 먼저** 호출돼야 합니다. 이유: archer의 C++ `ExpertDispatcher`가 static init 때 `torch.cuda.device_count()`를 읽어서, GPU가 N개로 보이면 rank마다 N개의 스레드를 띄워버립니다. `CUDA_VISIBLE_DEVICES`로 1개만 보이게 먼저 막아야 rank당 1 GPU·1 스레드가 됩니다.

### 2.3 NCCL 프로세스 그룹 (distributed_setup.py)

`init_distributed(ep_size)`에서:

```python
torch.cuda.set_device(0)                     # rank당 보이는 유일한 GPU
dist.init_process_group(
    backend="nccl", init_method="env://",
    timeout=_td(minutes=120),                # cold-start 30분+ 대비
)
```

- backend는 **NCCL**, rendezvous는 `env://` (torchrun이 채운 변수 사용).
- timeout 120분: 콜드스타트(모델 materialize ~30분) + warmup barrier에서 느린 rank 대기 대비.
- **EP=DP=world_size 강제**: `config.resolve_parallel()`과 `init_distributed()`가 `ep_size != world_size`면 즉시 에러. 즉 별도 DP 그룹 없이 **모든 rank가 하나의 EP 그룹**이고, 각 rank가 자기 배치 샤드를 가집니다 (암묵적 DP=world_size).

### 2.4 NUMA 공유 호스트 메모리 — "모델을 NUMA당 한 번만 올린다"

같은 NUMA에 GPU가 여러 개여도, **전문가 weight를 호스트RAM에 rank마다 복제하면 RAM이 터집니다.** 그래서:

1. NUMA 그룹에서 **가장 작은 local_rank가 leader**가 됩니다 ([entry.py:230](../moe_infinity_ep/launch/entry.py#L230) 근처).
2. leader가 `memfd_create`로 익명 RAM 영역을 만들고, 디스크에서 전문가 weight를 거기로 적재합니다.
3. follower들은 Unix 소켓(`/tmp/moe_numa{N}.sock`)으로 접속해 **`SCM_RIGHTS`로 fd를 건네받아** 같은 물리 페이지를 `mmap`합니다 (커널 페이지 dedup → 물리 RAM은 1벌).
4. follower는 디스크 읽기를 **skip** (`MOE_INFINITY_NUMA_FOLLOWER=1`)하지만, archer의 allocation은 leader와 **같은 순서로 lockstep** 진행해 같은 offset에 매핑됩니다.

> **★ 모든 rank가 각자 `cudaHostRegister`를 해야 한다 (롤백 금지 결정).** 물리 RAM은 공유지만, CUDA의 "이 호스트 메모리는 pinned" 등록 테이블은 **프로세스(컨텍스트)마다 별도**입니다. leader만 등록하고 follower가 skip하면, follower의 H2D 복사가 느린 pageable bounce-buffer 경로로 떨어집니다. 그래서 leader·follower **전원**이 `cudaHostRegister(PORTABLE|MAPPED)`를 호출합니다 ([numa_shared_host.py:102](../moe_infinity_ep/launch/numa_shared_host.py#L102) 부근). 실측 오버헤드는 cgroup +0.6%뿐이고 GPU util은 전 rank 90–92%로 대칭 유지.

### 2.5 적재 완료 동기화

`engine.load()` 끝에서 `post_init_barrier`로 전 rank가 만납니다. follower는 여기서 leader의 sparse-load가 memfd에 다 보일 때까지 대기합니다. barrier에 `BEGIN`만 찍혀 있으면 = 어떤 peer가 도착 못 함 (그 peer의 마지막 stage 로그를 보면 됨, `grep mi-init`).

---

## 3. coslot의 심장: 한 MoE 레이어가 실행되는 전체 흐름

이제 모델이 떴습니다. `model.generate()`가 도는 동안, **MoE 레이어 하나하나**가 어떻게 처리되는지가 이 시스템의 핵심입니다.

### 3.1 진입: HF 모듈 → executor

HuggingFace의 `Qwen3MoeSparseMoeBlock`이 우리 `Qwen3MoEBlockEP`로 교체돼 있습니다. 그 `forward()`가 하는 일은 단순합니다 ([qwen_ep.py:26](../moe_infinity_ep/models/qwen_ep.py#L26)):

```python
hidden = hidden.view(N, H)                    # [batch, seq, H] → [N, H]
is_decode = (sequence_length == 1)            # ★ decode/prefill 판별은 여기
final = self.expert_executor.run_layer(
    layer_id, hidden, gate, lib, is_decode, batch_size, sequence_length)
```

`run_layer`는 ep_size로 분기합니다 ([ep_executor.py:104](../moe_infinity_ep/runtime/ep_executor.py#L104)):
- `ep_size == 1` → `_single_rank_dispatch` (NCCL 없음)
- `ep_size > 1` → `_coop_dispatch` (★ 진짜 EP 경로, 아래)

### 3.2 `_coop_dispatch`: 한 레이어의 6단계 ([ep_executor.py:174](../moe_infinity_ep/runtime/ep_executor.py#L174))

아래 흐름이 **레이어마다, decode 토큰마다** 반복됩니다.

```
router_mask [N, E]  (이 rank의 토큰들이 어떤 expert를 원하는가)
        │
 ┌──────┴───────────────────────────────────────────────────────────────┐
 │ Phase 0  begin_layer(layer_id)         이전 레이어 정합성(I1) 확인         │
 ├───────────────────────────────────────────────────────────────────────┤
 │ Phase 1  classify_and_kick_hit()       ★ NCCL all_gather (수요 수집)      │
 │          → 전역 수요로 hit/miss 분류, miss를 어느 rank가 맡을지 배정        │
 ├───────────────────────────────────────────────────────────────────────┤
 │ Phase 2  plan_misses()                 rank별 fetch/evict 순차 계획         │
 │          → shadow cache 즉시 갱신, 통합 expert→rank 라우팅표 생성          │
 ├───────────────────────────────────────────────────────────────────────┤
 │ A2A-FWD  pack_tokens + route_tokens    ★ all_to_all #1 (토큰을 주인에게)   │
 ├───────────────────────────────────────────────────────────────────────┤
 │ COMPUTE  set_inputs + submit_plan      ★ archer: fetch + GEMM + combine    │
 │          + wait_layer_done()           (fetch stream ↔ compute stream 겹침)│
 ├───────────────────────────────────────────────────────────────────────┤
 │ A2A-BWD  return_outputs + scatter      ★ all_to_all #2 (결과를 원래대로)   │
 ├───────────────────────────────────────────────────────────────────────┤
 │ Phase 4  verify_layer_end()            shadow == archer 실제 (drift==0)    │
 └───────────────────────────────────────────────────────────────────────┘
        │
final [N, H]
```

이제 각 단계를 **"누가 명령하고, 무엇이 행해지는지"** 관점에서 풉니다.

---

## 4. Phase 1 — 전역 수요 수집과 hit/miss 분류 (컨트롤러 두뇌 #1)

### 4.1 왜 all_gather가 필요한가

각 rank는 **자기 토큰**이 어떤 expert를 원하는지만 압니다. 하지만 "이 expert를 누가 갖고 있고(hit), 누가 새로 가져와야 하나(miss)"는 **전체 rank의 수요를 합쳐야** 결정할 수 있습니다. 그래서 첫 통신이 일어납니다.

`demand_collector.collect_with_count()` ([demand_collector.py:49](../moe_infinity_ep/controller/demand_collector.py#L49)):

```python
local_count = local_router_mask.sum(dim=0)   # 이 rank가 expert별로 몇 토큰 원하나
dist.all_gather(list(per_rank.unbind(0)), local_count, group=ep_group)
# 결과: per_rank[r, e] = rank r 가 expert e 를 원하는 토큰 수  ([ep_size, num_experts])
```

> **이것이 레이어당 첫 번째 집합 통신입니다.** a2a가 아니라 **all_gather**입니다 (수요 행렬 공유용). 계측에서는 `cache_sync_us`로 잡힙니다.

### 4.2 hit/miss 분류 (전 rank가 동일하게)

전역 수요 행렬을 모든 rank가 똑같이 받으므로, **모든 rank가 byte-identical한 결정**을 내립니다. 이것이 coslot의 핵심 불변식입니다 — 컨트롤러는 분산이 아니라, 모든 rank에 복제돼 같은 입력으로 같은 답을 냅니다.

`global_controller.classify_and_kick_hit()`이:
1. 전 rank가 원하는 expert들의 합집합(union)을 구하고,
2. 각 expert가 `cache_view.locate(key)`로 **누군가의 GPU 슬롯에 이미 있으면 hit**, 없으면 **miss**,
3. hit은 `touch_use()`로 LRU 메타(last_used/freq) 갱신 (슬롯 물리 이동 없음),
4. miss는 `owner_policy.assign_rank()`로 **"이 miss expert는 어느 rank가 fetch할지"** 배정.

### 4.3 owner policy — miss를 어느 rank에 줄까 ([owner_policy.py](../moe_infinity_ep/controller/owner_policy.py))

| 정책 | 규칙 | 장점 / 단점 |
|------|------|-------------|
| `StaticPlacement` ("naive") | `expert_id % ep_size`가 고정 home | 단순·캐시 locality 좋음 / hot expert면 한 rank만 과부하 |
| `BalancedPlacement` | 매 레이어 fetch 수가 균등하도록 greedy 배치 | rank별 fetch 부하 균등 / 같은 expert가 매번 다른 rank → locality 약화 |
| `DemandAwareOwner` | 토큰 수(demand) 가중 LPT greedy | 무거운 expert를 가벼운 rank로 / 가장 복잡 |

> **fetch-bound 관찰 (프로젝트 메모).** balanced의 목적은 PCIe fetch를 rank 간 균등화하는 것이지만, decode는 사실상 weight-copy bound이고 fetch 비중이 0.7% 수준이라 aggregate는 naive도 이미 균등합니다. balanced는 per-layer 편차만 개선합니다. 그래서 기본은 `naive`.

Phase 1은 끝나자마자 `Phase1Output`을 돌려줘서, executor가 곧바로 a2a 준비에 들어갈 수 있게 합니다.

---

## 5. Phase 2 — rank별 fetch 계획과 shadow cache (컨트롤러 두뇌 #2)

### 5.1 슬롯(slot)이란?

GPU HBM 안에 **고정 크기 전문가 칸**을 `cap`개 미리 잡아둡니다. 슬롯 하나 = expert 하나가 들어가는 물리 자리입니다. `cap`은 `expert_specs.py`의 설계로 정해집니다:

```
cap = ratio × HBM / per_expert_bytes,   단 0.9×HBM hard ceiling
```

- per-expert bytes = `3 × intermediate × hidden × 2(bf16)` (gate/up/down 3개 행렬). 예: 235B ≈ 36 MiB/expert, 30B ≈ 9 MiB/expert.
- 235B 권장 `sparse_hbm_ratio ≈ 0.4~0.5`.

각 rank의 슬롯 상태(`RankSlotCache`)는 **shadow cache**입니다 — archer의 실제 GPU 잔존 상태를 Python이 복제해 들고 있는 "그림자". 컨트롤러는 항상 이 그림자로 계획하고, Phase 4에서 실제와 대조(drift==0)해 어긋나면 즉시 죽습니다.

### 5.2 순차 계획 + 즉시 apply (핵심 트릭) ([rank_planner.py:42](../moe_infinity_ep/controller/rank_planner.py#L42))

각 rank의 miss 목록을 demand 내림차순으로 정렬한 뒤, 하나씩:

```python
for expert in ordered_misses:
    empty = rank_cache.first_empty_slot()
    if empty is not None:
        dst_slot, victim = empty, None         # 빈 칸이면 그냥 채움
    else:
        victim_slot = evict_policy.pick_victim(rank_cache, expert, demand)
        dst_slot, victim = victim_slot, rank_cache.slots[victim_slot]  # 쫓아낼 놈 선택

    fetch_ops.append(FetchOp(expert, rank, dst_slot, victim, order))
    rank_cache.apply(dst_slot, evict=victim, insert=expert)   # ★ 즉시 그림자 갱신
```

`apply`를 **매 expert마다 즉시** 하기 때문에, 다음 expert의 victim 선택이 갱신된 상태를 봅니다. → 같은 슬롯을 두 번 victim으로 고르거나, 방금 넣은 걸 다시 쫓아내는 낭비가 원천 차단됩니다. (그래서 LRU OrderedDict의 "tail move" 청소가 필요 없습니다 — 메타만 읽으면 항상 최신.)

### 5.3 결과물: `LayerPlan`

- `fetch_ops`: 전 rank의 모든 fetch (expert, fetcher_rank, dst_slot, victim, order)
- `expert_to_rank`: hit 주인 ∪ miss fetcher의 **통합 라우팅표** (hit/miss disjoint 단언)
- 이 표를 GPU 텐서 `[num_experts] → rank`로 만들어(`build_expert_rank_table`) 토큰 라우팅에 씁니다.

> **evict 정책**: LRU(last_used) / LFU(freq) / DemandAware(이번 레이어 수요 적은 놈 먼저) 세 가지. victim은 빈 칸이 없을 때만 고릅니다.

---

## 6. A2A #1 (Forward) — 토큰을 전문가 주인에게 보내기 ([nvlink_router.py](../moe_infinity_ep/exec/nvlink_router.py))

이제 "이 토큰은 expert 5를 원하고, expert 5는 rank 2가 처리한다"는 라우팅표가 있습니다. 그러면 **내 토큰을 그 토큰이 필요한 expert의 주인 rank로 실제로 보내야** 합니다. 이것이 첫 번째 a2a입니다.

### 6.1 pack_tokens — 보낼 짐 싸기 ([nvlink_router.py:179](../moe_infinity_ep/exec/nvlink_router.py#L179))

`router_mask`에서 (토큰, expert) 쌍을 뽑아, 각 쌍의 목적지 rank를 라우팅표로 찾고, **목적지 rank 순으로 정렬**해 연속 블록으로 만듭니다. 결과 `RouterPlan`:
- `send_hidden [K, H]`: 목적지 순으로 재배열된 토큰 hidden
- `send_counts [ep_size]`: rank별로 몇 개 보내는지
- `token_idx [K]`: 원래 위치 (나중에 결과 되돌릴 때 씀)

### 6.2 fast-a2a — count 교환 통신을 없애기

보통 a2a는 "내가 너한테 몇 개 보낼게"를 먼저 교환(`exchange_counts`, 1회 추가 collective + GPU sync)해야 합니다. 하지만 우리는 **이미 Phase 1에서 전역 수요 행렬을 갖고 있고**, 라우팅표도 결정론적이므로, 통신 없이 순수 Python 계산으로 send/recv count를 도출합니다 ([derive_split_counts](../moe_infinity_ep/exec/nvlink_router.py#L142)):

```python
send_counts[s] = Σ_{e: route(e)==s}   per_rank[my_rank][e]
recv_counts[r] = Σ_{e: route(e)==my} per_rank[r][e]
```

> **`MOE_EP_FAST_A2A=1` (기본 on).** collective 1개 + device sync 2개를 제거합니다. 단, 프로젝트 결정상 일부 경로(#1, #2)에만 적용. EP는 NCCL a2a가 지배적이라(decode a2a는 sync-bound) 이 최적화로 **약 −35%** 효과.

### 6.3 route_tokens — 진짜 all_to_all ([nvlink_router.py:334](../moe_infinity_ep/exec/nvlink_router.py#L334))

```python
# 기본: hidden | expert_idx | weights 를 한 버퍼에 byte-pack → 단일 collective
dist.all_to_all_single(
    recv_buf, send_buf,
    output_split_sizes=recv_counts,
    input_split_sizes=send_counts,
    group=ep_group,
)
```

세 가지(hidden, expert 인덱스, weight)를 한 버퍼에 묶어 **all_to_all_single 한 번**으로 보냅니다 (legacy는 3번). 끝나면 이 rank는 `recv_hidden [K_recv, H]` — **자기가 처리해야 할 토큰들**을 받습니다.

> **이름이 nvlink_router지만 실제 전송은 NCCL `all_to_all_single`** 입니다. NVLink는 NCCL이 토폴로지에 맞춰 알아서 쓰는 하드웨어 경로일 뿐, 직접 NVLink API를 부르진 않습니다.

---

## 7. COMPUTE — archer가 fetch하고 GEMM하고 합치기 (손발)

받은 토큰(`recv_hidden`)을 실제 전문가 행렬과 곱해야 합니다. 그런데 그 전문가 weight가 GPU에 없을 수도 있습니다(miss). 여기서 **Python은 명령만 내리고, archer C++가 fetch+compute를 수행**합니다.

### 7.1 입력 세팅 + 단일 명령

```python
# recv 토큰을 archer 입력으로 (행마다 정확히 1개의 (token,expert) 쌍)
self.local_dispatcher.set_inputs(recv_hidden, recv_router_mask, recv_weights, is_decode)

# ★ 이번 레이어 계획표를 archer에 통째로 제출 (유일한 명령 통로)
self.controller.command_dispatcher.submit_plan(plan, hit_ops, ep_rank)

# archer가 fetch+GEMM+combine 끝낼 때까지 블록
final_local = self.local_dispatcher.wait_layer_done()
```

`command_dispatcher.submit_plan()` ([command_dispatcher.py:67](../moe_infinity_ep/controller/command_dispatcher.py#L67))는 전역 계획을 **이 rank 몫만 필터**해서 넘깁니다:
- `hit_tuples`: `(layer, expert, slot)` — 이미 GPU에 있는 것 (fetch 불필요, 조회만)
- `miss_tuples`: `(layer, expert, dst_slot, victim_layer, victim_expert, order)` — 가져올 것

```python
self.local_dispatcher.submit_plan(0, hit_tuples, miss_tuples)   # archer C++ 진입
```

> rank-local `order`는 반드시 `0..n-1` 연속이어야 합니다 — archer의 **PlanQueue가 FIFO**라서 이 순서대로 처리하기 때문 ([command_dispatcher.py:98](../moe_infinity_ep/controller/command_dispatcher.py#L98)).

### 7.2 archer 내부에서 일어나는 일 (PlanQueue)

archer(별도 `-archer-coslot` 레포, `submit_plan` 지원)는 받은 계획표를 PlanQueue로 처리합니다:
1. miss마다: victim이 있으면 그 슬롯을 비우고, **expert weight를 호스트RAM(또는 디스크)에서 `dst_slot`으로 H2D 복사(fetch)**. 슬롯이 GEMM 중이면 staging 슬롯(+1 예비 칸)에 먼저 받았다가 안전해지면 D2D commit.
2. hit + fetch 완료된 expert 전부에 대해 **GEMM 실행**.
3. partial들을 합쳐(combine) `final_local [K_recv, H]` 생성.

여기서 **이 시스템의 성능 비밀, compute/fetch 스트림 overlap**이 일어납니다 → §9에서 따로 자세히.

---

## 8. A2A #2 (Backward) — 결과를 원래 토큰 자리로 되돌리기

archer가 만든 `final_local`은 "내가 처리한 토큰들의 결과"입니다. 이걸 **원래 그 토큰을 보낸 rank로 되돌려** 줘야 합니다. 두 번째 a2a입니다.

```python
send_back = return_outputs(final_local, recv_counts, send_counts, ep_group)  # all_to_all_single
scatter_combine_into(final, send_back, rplan.token_idx, dtype)               # 원래 위치에 index_add_
```

`return_outputs` ([nvlink_router.py:404](../moe_infinity_ep/exec/nvlink_router.py#L404))는 forward a2a의 **정확한 역방향**(send/recv split을 바꿔치기)입니다. 받은 `send_back [K, H]`을 `token_idx`로 원래 `[N, H]` 자리에 scatter-add 하면 이 레이어의 최종 출력 완성.

> **★ 여기에 미묘한 cross-stream 버그가 숨어 있었습니다 (M5 fix).** `final_local`은 **archer의 스트림**에서 만들어졌는데, NCCL a2a는 **caller의 current stream**에서 돕니다. 아무 조치 없으면 caching allocator가 archer 스트림이 끝나자마자 `final_local` 메모리를 재활용해버려, NCCL이 읽기 전에 덮어써져 **조용히 NaN**이 됩니다. 그래서:
> ```python
> local_output.record_stream(torch.cuda.current_stream())  # 이 버퍼 NCCL이 쓸 거라고 allocator에 통보
> local_output = local_output.contiguous()                 # 다른 스트림 storage면 동기화 복사 강제
> ```
> 이게 Python 쪽에서 **유일하게 명시적인 cross-stream 동기화**입니다.

---

## 9. ★ compute stream과 fetch stream은 어떻게 서로를 가리나 (overlap)

질문의 핵심입니다. 정확히 짚으면 — **stream overlap의 실제 실행은 Python이 아니라 archer C++ 안에서 일어납니다.** Python 레포에는 명시적 `torch.cuda.Event()`나 `Stream()`이 (위의 record_stream 빼고) 없습니다. 대신 **어떻게 겹칠 수 있게 계획을 짜주는지**가 이 레포의 역할입니다.

### 9.1 두 종류의 overlap

**(A) 레이어 내부: fetch ↔ compute (archer 담당)**

archer는 최소 두 스트림을 운용합니다:
- **fetch stream**: miss expert weight의 H2D 복사 (PCIe)
- **compute stream**: GEMM 실행

PlanQueue는 hit expert(이미 GPU에 있음)의 GEMM을 compute stream에서 돌리는 **동안**, miss expert를 fetch stream에서 가져옵니다. fetch가 끝난 expert는 **cudaEvent gate**로 "복사 완료" 신호를 받은 뒤에 GEMM에 들어갑니다 (프로젝트 메모 *Streaming fetch: ExplicitFetchAsync + cudaEvent gate*). 즉:

```
compute stream:  [hit expert GEMM ......][fetched expert GEMM ...]
fetch   stream:  [miss A H2D][miss B H2D] ↑event
                                          └─ B 복사 완료 event를 GEMM이 wait
```

→ **PCIe 복사 시간이 GEMM 시간 뒤로 숨습니다(hidden).** 이게 offloading 시스템의 핵심 이득입니다.

불변식 **I5**: hit-compute의 read는 fetch-replace의 write보다 먼저 끝나야 합니다 (같은 슬롯을 읽는 중에 덮어쓰면 안 됨). staging 슬롯(+1 예비 칸)이 이걸 보장합니다 — 슬롯이 사용 중이면 예비 칸에 먼저 받고 나중에 commit.

**(B) 레이어 경계: 이전 레이어 compute ↔ 다음 레이어 fetch**

PlanQueue는 한 번에 하나의 active fetch를 돌리되, **이전 레이어의 combine이 끝나기 전에 다음 레이어의 fetch를 시작**할 수 있습니다 (스트림이 다르므로). 이로써 레이어 간 경계의 PCIe 시간도 가려집니다.

### 9.2 Python이 overlap을 "가능하게" 만드는 방법

archer가 겹칠 수 있으려면 **fetch 대상이 미리·정확히** 정해져 있어야 합니다. 그래서:
1. Phase 1의 all_gather로 전역 수요를 미리 알고,
2. Phase 2에서 **레이어 전체의 fetch 계획을 한 번에** 확정하고,
3. `submit_plan`으로 **통째로** 넘깁니다 (per-op 명령 왕복이 아님).

per-op으로 "하나 가져와, 하나 계산해"를 주고받으면 매번 Python↔C++ 왕복 + 동기화가 생겨 overlap이 깨집니다. **계획을 통째로 넘기는 coslot 설계 자체가 overlap의 전제**입니다.

### 9.3 NCCL 스트림과의 관계

a2a(NCCL)는 또 다른(current) 스트림에서 돕니다. 그래서 전체적으로 세 종류의 작업 — **NCCL a2a / archer fetch / archer compute** — 가 서로 다른 스트림에서 부분적으로 겹칩니다. 다만 우리 흐름은 `wait_layer_done()`에서 compute 완료를 블록 대기하므로, 한 레이어 안에서 a2a-fwd → compute → a2a-bwd는 순차적입니다. overlap은 주로 **archer 내부(fetch↔compute)와 레이어 경계**에서 발생합니다.

---

## 10. Phase 4 — 정합성 검증 (drift == 0)

레이어 끝에서 컨트롤러가 자기 그림자(shadow)와 archer의 실제 GPU 잔존 상태를 대조합니다 ([global_controller.py](../moe_infinity_ep/controller/global_controller.py) `verify_layer_end`):

```python
physical = set(local_dispatcher.get_cached_experts(0))   # archer의 실제 잔존
drift = verify_against_archer(physical)                   # shadow와 비교
if drift > 0: raise RuntimeError(...)                     # 즉시 죽음
cache_view.bump_all_layers()                              # 다음 레이어 준비
```

> 235B 실 GPU 검증에서 **drift=0, corruption 없음**을 확인 (프로젝트 메모 *Archer stream/lock verify*). logit noise는 cuBLAS 본질적인 것이지 버그가 아님.

---

## 11. decode vs prefill — 무엇이 다른가

컨트롤러 코드는 둘을 **구분하지 않습니다**. 차이는 입력 모양에서 자연히 나옵니다:

| | prefill | decode |
|---|---------|--------|
| `router_mask` | `[batch×seq, E]` (토큰 많음) | `[batch, E]` (시퀀스당 1토큰) |
| a2a 메시지 크기 | 큼 | 작음 |
| 병목 | a2a **imbalance**-bound | a2a **sync**-bound, weight-copy bound |
| `is_decode` 판별 | `seq_len > 1` | `seq_len == 1` ([qwen_ep.py:44](../moe_infinity_ep/models/qwen_ep.py#L44)) |

같은 6단계를 똑같이 돌지만, decode는 토큰이 적어 a2a가 작고 동기화 지연이 지배적이라 **fast-a2a의 이득이 큽니다**. 계측은 `_routed_decode`/`_routed_prefill`, `decode_layer_total_us`/`prefill_layer_total_us`로 분리 집계합니다.

---

## 12. 계측 — 무엇이 측정되나 ([instrument/](../moe_infinity_ep/instrument/))

매 레이어 `counters`에 누적:
- `cache_sync_us` — Phase 1 all_gather (수요 수집)
- `a2a_forward_us` / `a2a_backward_us` — 두 a2a 시간
- `local_exec_us` — archer fetch+GEMM+combine 시간
- `layer_hits / layer_misses / layer_evictions / layer_remote_serves`
- `dispatch_fetch_us` (p50/p99) — fetch 1회 지연 (235B expert ≈ 10ms PCIe)
- `drift_layer_pre` — shadow↔archer 불일치 (0이 정상)
- `cap_utilization_max`, `high_cap_pressure_layers` — 캐시 압박도

`trace.py`의 `gather_and_dump`이 전 rank를 rank0로 모아 JSON/CSV로 떨굽니다.

---

## 13. 전체 흐름 한눈에 (시퀀스)

```
[프로세스 기동]  torchrun → numa_wrap → pin GPU → NCCL init(120min) → NUMA memfd 공유
                + 전원 cudaHostRegister → leader 디스크적재 / follower skip → barrier
        │
        ▼  model.generate() 루프, 레이어마다:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Qwen3MoEBlockEP.forward → executor.run_layer → _coop_dispatch                  │
│                                                                               │
│  Phase0 begin_layer                                                            │
│  Phase1 ─── all_gather(수요) ───────────────► [NCCL #0]  hit/miss 분류, owner배정 │
│  Phase2 rank별 fetch 계획 + shadow 갱신 → expert→rank 표                         │
│  A2A-fwd pack_tokens → all_to_all_single ──► [NCCL #1]  토큰을 주인 rank로         │
│  COMPUTE set_inputs → submit_plan ─────────► [archer C++]                       │
│              └ PlanQueue: fetch(stream A) ∥ GEMM(stream B) ∥ combine            │
│              └ wait_layer_done() (블록)                                          │
│  A2A-bwd return_outputs → all_to_all_single ► [NCCL #2]  결과 원위치로            │
│              └ record_stream + contiguous (cross-stream 안전)                    │
│  Phase4 verify_layer_end (drift==0 아니면 즉사)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

레이어당 집합 통신은 **all_gather 1회 + all_to_all 2회 = 3회**입니다.

---

## 14. 자주 헷갈리는 점 정리

1. **"컨트롤러가 분산돼 있나?"** → 아니오. 모든 rank가 **동일한 컨트롤러 복제본**을 들고, 동일한 전역 수요로 동일한 결정을 내립니다. all_gather로 입력을 똑같이 맞추는 게 전부입니다.
2. **"nvlink_router가 NVLink를 직접 부르나?"** → 아니오. NCCL `all_to_all_single`을 부르고, NVLink는 NCCL이 알아서 쓰는 하드웨어 경로입니다.
3. **"stream overlap을 Python이 하나?"** → 아니오. Python은 **계획을 통째로 미리** 넘겨 archer가 fetch↔compute를 겹칠 수 있게만 합니다. 실제 스트림·event는 archer C++ 안에 있습니다 (Python 쪽 명시적 동기화는 return a2a의 `record_stream` 하나뿐).
4. **"a2a는 몇 번?"** → 레이어당 forward 1번(토큰 보내기) + backward 1번(결과 받기). 그 앞에 all_gather 1번(수요)이 별도로 있습니다.
5. **"fetch는 누가 정하나?"** → 컨트롤러(Phase 2)가 정확한 dst_slot·victim까지 다 정해서 넘기고, archer는 그대로 복사만 합니다 (archer 자율 evict는 기본 off, `MOE_EP_DISABLE_ARCHER_EVICT=1`).

---

## 15. 더 읽을거리 (이 레포의 다른 문서)

- `docs/controller_flow_story.md` — 컨트롤러 흐름 서사
- `docs/controller_flow_verification.md` — 불변식 검증 상세
- `docs/ep_offloading_flow.md` — EP offloading 흐름
- `docs/expert_cache_mechanism.md` — 캐시 메커니즘 심화
- `revision.md` — coslot 재작성 변경 이력
```
