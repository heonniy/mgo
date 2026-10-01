// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// 2026-05-29 Controller-Owned Slot Execution (coslot) rewrite — see the
// header for the high-level design.  revision.md Phases 2-7 are the spec.

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "model/model_topology.h"
#include "model/moe.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <chrono>

// sentinel for an empty slot.  Can't use 0 because key(layer=0, expert=0) == 0.
static constexpr uint64_t kSlotEmpty = ~0ULL;

// env-gated step trace through the coslot pipeline (revision.md §14).  Off by
// default (zero cost); MOE_EP_EXEC_TRACE=1 prints one line per stage.
static bool IsExecTraceEnabled() {
  static const bool kEnabled = []() {
    const char* env = std::getenv("MOE_EP_EXEC_TRACE");
    return env != nullptr && std::string(env) == "1";
  }();
  return kEnabled;
}
#define CO_LOG(...)                                                  \
  do {                                                              \
    if (IsExecTraceEnabled()) {                                     \
      fprintf(stderr, __VA_ARGS__);                                 \
      fflush(stderr);                                               \
    }                                                               \
  } while (0)

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype,
                                   int expert_type, int num_threads)
    : num_experts_(num_experts),
      dtype_(dtype),
      expert_type_(expert_type),
      modules_(kNumDevices(), nullptr),
      slot_pool_base_(kNumDevices(), nullptr),
      slot_device_ptr_(kNumDevices()),
      slot_to_key_(kNumDevices()),
      key_to_slot_(kNumDevices()),
      cached_experts_(kNumDevices()),
      cache_sizes_(),
      slot_last_compute_event_(kNumDevices()),
      sched_mutex_(kNumDevices()),
      plan_queue_(kNumDevices()),
      has_active_fetch_(kNumDevices(), 0),
      active_parked_(kNumDevices(), 0),
      active_op_(kNumDevices()),
      expected_order_(kNumDevices(), 0),
      slot_state_(kNumDevices()),
      snapshots_(kNumDevices()),
      partial_outputs_(kNumDevices()),
      partial_seq_(kNumDevices(), 0),
      exec_queue_(kNumDevices()),
      h2d_streams_(kNumDevices(), nullptr),
      pending_(0) {
  main_thread_stop_flag_.store(false);

  // Per-GPU dedicated H2D streams (non-blocking so they overlap exec/NCCL).
  for (int i = 0; i < kNumDevices(); ++i) {
    cudaSetDevice(i);
    auto err =
        cudaStreamCreateWithFlags(&h2d_streams_[i], cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      DLOG_FATAL("ExpertDispatcher: cudaStreamCreate h2d_stream[", i,
                 "] failed: ", cudaGetErrorString(err));
    }
  }

  // EXACTLY one exec worker per GPU (revision.md §5.1).  partial_outputs_ /
  // slot_state_ have a single writer.  num_threads is ignored (coslot
  // invariant); log + FATAL if a caller ever expects >1.
  int workers_per_gpu = 1;
  if (num_threads != 1) {
    DLOG_WARN("[CO_SLOT] num_threads=", num_threads,
              " ignored; exec_workers_per_gpu forced to 1");
  }
  fprintf(stderr, "[CO_SLOT] exec_workers_per_gpu=%d\n", workers_per_gpu);
  fflush(stderr);

  for (int i = 0; i < kNumDevices(); ++i) {
    cudaSetDevice(i);
    cudaStream_t exec_stream;
    cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
    exec_streams_.push_back(exec_stream);

    auto cache_limit =
        kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, i));
    cache_sizes_.push_back(cache_limit);

    modules_[i] = new MoEMLP(dtype, expert_type);

    auto thread_func = std::bind(&ExpertDispatcher::GPUExecFunc, this, i);
    std::string thread_name = "GPUExecFunc" + std::to_string(i);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
  }

  at::InferenceMode infer_guard(0);
  for (int i = 0; i < num_experts; ++i) {
    experts_.emplace_back();
    for (int j = 0; j < num_layers; ++j) {
      experts_[i].emplace_back();
      experts_[i][j] = std::make_shared<ExpertNode>();
      experts_[i][j]->expert_type = expert_type;
      switch (expert_type_) {
        case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
          experts_[i][j]->module = new SwitchTransformersDenseActDense(dtype);
          break;
        case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
          experts_[i][j]->module =
              new SwitchTransformersDenseGatedActDense(dtype);
          break;
        case NLLB_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new NllbMoeDenseActDense(dtype);
          break;
        case FSGPT_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new FSGPTMoEDenseActDense(dtype);
          break;
        case MIXTRAL_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new MixtralMoEDenseActDense(dtype);
          break;
        case DEEPSEEK_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new DeepSeekMoEDenseActDense(dtype);
          break;
        default:
          DLOG_FATAL("ExpertDispatcher: unknown expert type ", expert_type_);
      }
      experts_[i][j]->module->eval();
      experts_[i][j]->layer_idx = j;
      experts_[i][j]->expert_idx = i;
    }
  }
}

ExpertDispatcher::~ExpertDispatcher() {
  main_thread_stop_flag_.store(true);
  // Wake the exec workers blocked on exec_queue_.Pop with a stop sentinel.
  for (int i = 0; i < kNumDevices(); ++i) {
    ExecTask stop;
    stop.stop = true;
    exec_queue_[i].Push(stop);
  }
  for (auto& thread : threads_) {
    thread->join();
  }
  for (auto& stream : exec_streams_) {
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
  for (auto& stream : h2d_streams_) {
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
}

void ExpertDispatcher::RegisterExpert(
    int layer_idx, int expert_idx, const std::vector<std::uint32_t>& tensor_ids,
    std::string jit_path) {
  NodePtr cached_node = nullptr;
  for (auto tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (cached_node == nullptr) {
      cached_node = node;
      experts_[expert_idx][layer_idx]->node = node;
    } else if (cached_node != node) {
      DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
    }
  }
}

void ExpertDispatcher::SetInputs(const torch::Tensor& hidden_states,
                                 const torch::Tensor& router_mask,
                                 const torch::Tensor& router_weight,
                                 bool is_decode) {
  hidden_states_ = hidden_states;
  router_mask_ = router_mask;
  router_weight_ = router_weight;
  is_decode_.store(is_decode, std::memory_order_relaxed);
}

// ===================================================================
// Slot pool — (cap + 1) fixed slots; index `cap` is the staging slot.
// ===================================================================
void ExpertDispatcher::InitSlotPool(int cap_per_gpu, int64_t expert_byte_size) {
  if (slot_pool_initialized_) {
    DLOG_WARN("InitSlotPool called twice — ignoring");
    return;
  }
  if (expert_byte_size <= 0) {
    int64_t max_bytes = 0;
    for (auto& per_expert : experts_) {
      for (auto& en : per_expert) {
        if (en && en->node && en->node->byte_size > max_bytes) {
          max_bytes = en->node->byte_size;
        }
      }
    }
    if (max_bytes <= 0) {
      DLOG_FATAL("InitSlotPool: auto-detect found no registered experts with "
                 "byte_size>0.  Call after model load / RegisterExpert.");
    }
    expert_byte_size = max_bytes;
    DLOG_INFO("InitSlotPool: auto-detected expert_byte_size=", expert_byte_size);
  }
  slot_cap_ = cap_per_gpu;
  slot_expert_byte_size_ = expert_byte_size;
  int n = kNumDevices();
  int total_slots = cap_per_gpu + 1;  // +1 reserved staging slot (index cap)
  for (int g = 0; g < n; ++g) {
    cudaSetDevice(g);
    void* base = nullptr;
    size_t total = (size_t)total_slots * (size_t)expert_byte_size;
    auto err = cudaMalloc(&base, total);
    if (err != cudaSuccess) {
      DLOG_FATAL("InitSlotPool: cudaMalloc ", total, " bytes gpu ", g,
                 " failed: ", cudaGetErrorString(err));
    }
    slot_pool_base_[g] = base;
    slot_device_ptr_[g].assign(total_slots, nullptr);
    slot_last_compute_event_[g].assign(total_slots, nullptr);
    for (int s = 0; s < total_slots; ++s) {
      slot_device_ptr_[g][s] =
          static_cast<char*>(base) + (size_t)s * (size_t)expert_byte_size;
      auto ee = cudaEventCreateWithFlags(&slot_last_compute_event_[g][s],
                                         cudaEventDisableTiming);
      if (ee != cudaSuccess) {
        DLOG_FATAL("InitSlotPool: cudaEventCreate gpu ", g, " slot ", s,
                   " failed: ", cudaGetErrorString(ee));
      }
    }
    slot_to_key_[g].assign(cap_per_gpu, kSlotEmpty);  // resident slots only
    key_to_slot_[g].clear();
    cached_experts_[g].clear();
    slot_state_[g].assign(cap_per_gpu, FREE);
  }
  slot_pool_initialized_ = true;
  DLOG_INFO("InitSlotPool: cap=", cap_per_gpu, " (+1 staging)",
            " expert_byte_size=", expert_byte_size);
}

// ===================================================================
// Snapshot — token rows + weights for one expert (revision.md §6.3).
// Computed at SubmitPlan so the GEMM never re-reads the shared router_mask_.
// ===================================================================
void ExpertDispatcher::ComputeSnapshot(int expert, torch::Tensor* token_idx,
                                       torch::Tensor* weights,
                                       int64_t* token_count) {
  auto mask_col = router_mask_.select(1, expert).to(torch::kBool);  // [K_recv]
  auto idx = mask_col.nonzero().squeeze(1).contiguous();            // [K_e] long
  auto w = router_weight_.select(1, expert)
               .index_select(0, idx)
               .to(torch::kFloat32)
               .contiguous();                                       // [K_e]
  *token_idx = idx;
  *weights = w;
  *token_count = idx.size(0);
}

// ===================================================================
// SubmitPlan (revision.md §3.1 entry + Phase 5 hit push)
// ===================================================================
void ExpertDispatcher::SubmitPlan(
    int gpu,
    const std::vector<std::tuple<int, int, int>>& hit_ops,
    const std::vector<std::tuple<int, int, int, int, int, int>>& miss_ops) {
  if (!slot_pool_initialized_) {
    DLOG_FATAL("SubmitPlan: slot pool not initialized — call init_slot_pool().");
  }
  std::lock_guard<std::mutex> lk(sched_mutex_[gpu]);

  // ---- reset per-layer scheduler state ----
  partial_outputs_[gpu].clear();
  partial_seq_[gpu] = 0;
  snapshots_[gpu].clear();
  plan_queue_[gpu].clear();
  has_active_fetch_[gpu] = 0;
  active_parked_[gpu] = 0;
  expected_order_[gpu] = 0;

  int pending_local = 0;

  // ---- hits: validate slot, mark COMPUTING, snapshot + push ExecTask ----
  for (auto& h : hit_ops) {
    int layer = std::get<0>(h), expert = std::get<1>(h), slot = std::get<2>(h);
    uint64_t key = Key(layer, expert);
    if (slot < 0 || slot >= slot_cap_) {
      DLOG_FATAL("SubmitPlan HIT: slot ", slot, " OOB (cap=", slot_cap_,
                 ") L=", layer, " E=", expert);
    }
    if (slot_to_key_[gpu][slot] != key) {
      DLOG_FATAL("SubmitPlan HIT mismatch: slot ", slot, " holds ",
                 (int)(slot_to_key_[gpu][slot] >> 32), ",",
                 (int)(slot_to_key_[gpu][slot] & 0xFFFFFFFF),
                 " controller says hit (", layer, ",", expert, ")");
    }
    torch::Tensor tidx, w;
    int64_t cnt = 0;
    ComputeSnapshot(expert, &tidx, &w, &cnt);
    if (cnt == 0) {
      DLOG_FATAL("SubmitPlan HIT: token_count==0 for (", layer, ",", expert,
                 ") — controller must filter token-0 experts.");
    }
    ExecTask snap;
    snap.layer = layer;
    snap.expert = expert;
    snap.token_idx = tidx;
    snap.weights = w;
    snapshots_[gpu][key] = snap;
    slot_state_[gpu][slot] = COMPUTING;
    cache_hit_cnt_.fetch_add(1, std::memory_order_relaxed);
    PushExecTask(gpu, layer, expert, slot, /*fetch_event=*/nullptr);
    pending_local++;
  }

  // ---- misses: build PlanQueue (FIFO by order) + snapshots ----
  std::vector<PlanOp> ops;
  ops.reserve(miss_ops.size());
  for (auto& m : miss_ops) {
    PlanOp op;
    op.layer = std::get<0>(m);
    op.expert = std::get<1>(m);
    op.dst_slot = std::get<2>(m);
    op.victim_layer = std::get<3>(m);
    op.victim_expert = std::get<4>(m);
    op.order = std::get<5>(m);
    if (op.dst_slot < 0 || op.dst_slot >= slot_cap_) {
      DLOG_FATAL("SubmitPlan MISS: dst_slot ", op.dst_slot, " OOB (cap=",
                 slot_cap_, ") L=", op.layer, " E=", op.expert);
    }
    torch::Tensor tidx, w;
    int64_t cnt = 0;
    ComputeSnapshot(op.expert, &tidx, &w, &cnt);
    if (cnt == 0) {
      DLOG_FATAL("SubmitPlan MISS: token_count==0 for (", op.layer, ",",
                 op.expert, ") — controller must filter token-0 experts.");
    }
    ExecTask snap;
    snap.layer = op.layer;
    snap.expert = op.expert;
    snap.token_idx = tidx;
    snap.weights = w;
    snapshots_[gpu][Key(op.layer, op.expert)] = snap;
    ops.push_back(op);
    cache_miss_cnt_.fetch_add(1, std::memory_order_relaxed);
    pending_local++;
  }
  // sort by order (defensive; controller already emits in order).
  std::sort(ops.begin(), ops.end(),
            [](const PlanOp& a, const PlanOp& b) { return a.order < b.order; });
  for (auto& op : ops) plan_queue_[gpu].push_back(op);

  // FRAGILE INVARIANT: this store happens AFTER the hit loop already
  // PushExecTask'd (above), so an exec worker may already be popping a hit
  // task.  It is only safe because SubmitPlan holds sched_mutex_[gpu] for the
  // whole function, and GPUExecFunc must acquire sched_mutex_[gpu] (its slot
  // fail-fast check) before it can reach pending_.fetch_sub.  That ordering
  // guarantees this store happens-before any decrement of this layer's tasks,
  // so pending_ cannot underflow.  If the exec worker's pre-decrement mutex
  // acquire is ever removed/reordered, this store must move ABOVE the hit
  // PushExecTask loop (store pending_local before pushing any task).
  pending_.store(pending_local);
  CO_LOG("[SubmitPlan] gpu=%d num_hit=%zu num_miss=%zu pending=%d\n", gpu,
         hit_ops.size(), miss_ops.size(), pending_local);

  // kick the fetch pipeline (drains all DIRECT ops synchronously; parks on the
  // first STAGING op).
  StartNextFetch(gpu);
}

// ===================================================================
// Scheduler (revision.md §3-4) — all callers hold sched_mutex_[gpu].
// ===================================================================
void ExpertDispatcher::StartNextFetch(int gpu) {
  while (!has_active_fetch_[gpu] && !plan_queue_[gpu].empty()) {
    PlanOp op = plan_queue_[gpu].front();
    plan_queue_[gpu].pop_front();
    if (op.order != expected_order_[gpu]) {
      DLOG_FATAL("[CO_SLOT] PlanQueue order mismatch gpu=", gpu, " got ",
                 op.order, " expected ", expected_order_[gpu], " (E=", op.expert,
                 " L=", op.layer, ")");
    }
    expected_order_[gpu]++;
    has_active_fetch_[gpu] = 1;
    active_op_[gpu] = op;
    active_parked_[gpu] = 0;
    if (slot_state_[gpu][op.dst_slot] == FREE) {
      DoDirectFetch(gpu, op);   // commit + push; clears has_active_fetch_
    } else {
      DoStagingFetchPark(gpu, op);  // H2D->staging; parks (loop exits)
    }
  }
}

void ExpertDispatcher::DoDirectFetch(int gpu, const PlanOp& op) {
  auto node = experts_[op.expert][op.layer]->node;
  if (node == nullptr || node->host_memory_ptr == nullptr) {
    DLOG_FATAL("[CO_SLOT] DoDirectFetch: null node/host_memory_ptr (", op.layer,
               ",", op.expert, ")");
  }
  void* dst = slot_device_ptr_[gpu][op.dst_slot];
  CO_LOG("[FetchStart] order=%d E=%d dst_slot=%d victim=(%d,%d) mode=direct\n",
         op.order, op.expert, op.dst_slot, op.victim_layer, op.victim_expert);

  // gate slot overwrite on the prior occupant's GEMM (no-op for empty/stale).
  cudaStreamWaitEvent(h2d_streams_[gpu],
                      slot_last_compute_event_[gpu][op.dst_slot], 0);
  // alias the expert's torch tensors over the slot, then async H2D.
  node->device_memory_ptr = dst;
  node->device = CUDA_DEVICE(gpu);
  experts_[op.expert][op.layer]->SetTensorsFromBlob(CUDA_DEVICE(gpu));
  CudaMemcpyAsync(dst, node->host_memory_ptr, node->byte_size,
                  cudaMemcpyHostToDevice, h2d_streams_[gpu]);
  if (node->fetch_event == nullptr) {
    cudaSetDevice(gpu);
    auto e = cudaEventCreateWithFlags(&node->fetch_event,
                                      cudaEventDisableTiming);
    if (e != cudaSuccess) {
      DLOG_FATAL("DoDirectFetch: cudaEventCreate failed: ",
                 cudaGetErrorString(e));
    }
  }
  cudaEventRecord(node->fetch_event, h2d_streams_[gpu]);

  Commit(gpu, op);                       // victim/empty bookkeeping + FATAL
  slot_state_[gpu][op.dst_slot] = COMPUTING;  // a GEMM is now pending here
  PushExecTask(gpu, op.layer, op.expert, op.dst_slot, node->fetch_event);
  gpu_fetch_cnt_.fetch_add(1, std::memory_order_relaxed);
  direct_fetch_cnt_.fetch_add(1, std::memory_order_relaxed);
  has_active_fetch_[gpu] = 0;            // loop continues to next op
}

void ExpertDispatcher::DoStagingFetchPark(int gpu, const PlanOp& op) {
  auto node = experts_[op.expert][op.layer]->node;
  if (node == nullptr || node->host_memory_ptr == nullptr) {
    DLOG_FATAL("[CO_SLOT] DoStagingFetchPark: null node/host_memory_ptr (",
               op.layer, ",", op.expert, ")");
  }
  void* staging = slot_device_ptr_[gpu][slot_cap_];  // reserved staging slot
  CO_LOG("[FetchStart] order=%d E=%d dst_slot=%d victim=(%d,%d) mode=staging\n",
         op.order, op.expert, op.dst_slot, op.victim_layer, op.victim_expert);
  // Issue the expensive H2D into staging NOW so it overlaps the dst_slot's
  // in-flight GEMM.  The cheap D2D + commit are deferred to
  // CompleteStagingFetch (triggered when that GEMM finishes and records
  // slot_last_compute_event), so the slot's compute is provably done first.
  CudaMemcpyAsync(staging, node->host_memory_ptr, node->byte_size,
                  cudaMemcpyHostToDevice, h2d_streams_[gpu]);
  active_parked_[gpu] = 1;  // has_active_fetch_ stays 1 → StartNextFetch exits
}

void ExpertDispatcher::CompleteStagingFetch(int gpu) {
  const PlanOp& op = active_op_[gpu];
  auto node = experts_[op.expert][op.layer]->node;
  void* dst = slot_device_ptr_[gpu][op.dst_slot];
  void* staging = slot_device_ptr_[gpu][slot_cap_];
  // dst_slot's GEMM has finished and recorded slot_last_compute_event; gate
  // the D2D overwrite on it (now guaranteed already-recorded → effective).
  cudaStreamWaitEvent(h2d_streams_[gpu],
                      slot_last_compute_event_[gpu][op.dst_slot], 0);
  node->device_memory_ptr = dst;
  node->device = CUDA_DEVICE(gpu);
  experts_[op.expert][op.layer]->SetTensorsFromBlob(CUDA_DEVICE(gpu));
  CudaMemcpyAsync(dst, staging, node->byte_size, cudaMemcpyDeviceToDevice,
                  h2d_streams_[gpu]);
  if (node->fetch_event == nullptr) {
    cudaSetDevice(gpu);
    cudaEventCreateWithFlags(&node->fetch_event, cudaEventDisableTiming);
  }
  cudaEventRecord(node->fetch_event, h2d_streams_[gpu]);

  Commit(gpu, op);
  slot_state_[gpu][op.dst_slot] = COMPUTING;
  PushExecTask(gpu, op.layer, op.expert, op.dst_slot, node->fetch_event);
  gpu_fetch_cnt_.fetch_add(1, std::memory_order_relaxed);
  staging_fetch_cnt_.fetch_add(1, std::memory_order_relaxed);
  has_active_fetch_[gpu] = 0;
  active_parked_[gpu] = 0;
  StartNextFetch(gpu);  // continue draining the PlanQueue
}

void ExpertDispatcher::Commit(int gpu, const PlanOp& op) {
  uint64_t new_key = Key(op.layer, op.expert);
  if (op.victim_layer >= 0 && op.victim_expert >= 0) {
    uint64_t vkey = Key(op.victim_layer, op.victim_expert);
    if (slot_to_key_[gpu][op.dst_slot] != vkey) {
      uint64_t held = slot_to_key_[gpu][op.dst_slot];
      DLOG_FATAL("[CO_SLOT] VICTIM MISMATCH gpu=", gpu, " dst_slot=",
                 op.dst_slot, " controller_victim=(", op.victim_layer, ",",
                 op.victim_expert, ") archer_holds=(", (int)(held >> 32), ",",
                 (int)(held & 0xFFFFFFFF), ") expert=(", op.layer, ",",
                 op.expert, ") order=", op.order);
    }
    key_to_slot_[gpu].erase(vkey);
    cached_experts_[gpu].erase(vkey);
    auto vnode = experts_[op.victim_expert][op.victim_layer]->node;
    if (vnode != nullptr) {
      vnode->device_memory_ptr = nullptr;  // detach (slot memory reused)
      vnode->device = CPU_DEVICE;
      cache_sizes_[gpu] += vnode->byte_size;
    }
    eviction_cnt_.fetch_add(1, std::memory_order_relaxed);
  } else {
    if (slot_to_key_[gpu][op.dst_slot] != kSlotEmpty) {
      uint64_t held = slot_to_key_[gpu][op.dst_slot];
      DLOG_FATAL("[CO_SLOT] EMPTY-SLOT MISMATCH gpu=", gpu, " dst_slot=",
                 op.dst_slot, " expected empty, archer_holds=(",
                 (int)(held >> 32), ",", (int)(held & 0xFFFFFFFF),
                 ") expert=(", op.layer, ",", op.expert, ") order=", op.order);
    }
  }
  slot_to_key_[gpu][op.dst_slot] = new_key;
  key_to_slot_[gpu][new_key] = op.dst_slot;
  cached_experts_[gpu].insert(new_key);
  auto nnode = experts_[op.expert][op.layer]->node;
  if (nnode != nullptr) cache_sizes_[gpu] -= nnode->byte_size;
  CO_LOG("[FetchCommit] E=%d dst_slot=%d\n", op.expert, op.dst_slot);
}

void ExpertDispatcher::PushExecTask(int gpu, int layer, int expert, int slot,
                                    cudaEvent_t fetch_event) {
  auto it = snapshots_[gpu].find(Key(layer, expert));
  if (it == snapshots_[gpu].end()) {
    DLOG_FATAL("[CO_SLOT] PushExecTask: missing snapshot for (", layer, ",",
               expert, ")");
  }
  ExecTask t = it->second;  // copies token_idx/weights (shallow tensor refs)
  t.layer = layer;
  t.expert = expert;
  t.slot = slot;
  t.fetch_event = fetch_event;
  t.stop = false;
  exec_queue_[gpu].Push(t);
}

// ===================================================================
// Exec worker (revision.md §5) — one per GPU.
// ===================================================================
void ExpertDispatcher::GPUExecFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream = exec_streams_[gpu_id];

  while (!main_thread_stop_flag_.load()) {
    ExecTask task;
    exec_queue_[gpu_id].Pop(task);
    if (task.stop) break;
    if (task.expert < 0) continue;

    auto node = experts_[task.expert][task.layer]->node;

    // Fail-fast #2: the slot must currently hold this expert.
    {
      std::lock_guard<std::mutex> lk(sched_mutex_[gpu_id]);
      if (slot_to_key_[gpu_id][task.slot] != Key(task.layer, task.expert)) {
        uint64_t held = slot_to_key_[gpu_id][task.slot];
        DLOG_FATAL("[CO_SLOT] ExecTask slot mismatch gpu=", gpu_id, " slot=",
                   task.slot, " holds=(", (int)(held >> 32), ",",
                   (int)(held & 0xFFFFFFFF), ") expert=(", task.layer, ",",
                   task.expert, ")");
      }
    }
    CO_LOG("[ExecPop] E=%d slot=%d token_count=%ld\n", task.expert, task.slot,
           (long)task.token_idx.size(0));

    // Gate the slot->param_ copy (done inside SetTensorsFromIds on the null
    // stream) on the H2D/D2D completion.  cudaEventSynchronize is CPU-side;
    // by the time we pop a task its fetch usually finished (fetch ‖ prior
    // GEMM), so this rarely stalls.
    auto _tf0 = std::chrono::steady_clock::now();
    if (task.fetch_event != nullptr) {
      cudaEventSynchronize(task.fetch_event);
    }
    auto _tf1 = std::chrono::steady_clock::now();
    fetch_wait_us_.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(_tf1 - _tf0)
            .count(),
        std::memory_order_relaxed);

    // weight_copy = slot->param_ D2D weight copy (per expert, ~expert_bytes).
    // Timed separately from the GEMM so "compute" is not inflated by the
    // per-expert weight movement (the two were previously conflated).
    modules_[gpu_id]->SetTensorsFromIds(node->tensor_ids);
    auto _tw1 = std::chrono::steady_clock::now();
    weight_copy_us_.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(_tw1 - _tf1)
            .count(),
        std::memory_order_relaxed);

    // compute = the actual MoEMLP GEMM (forward internally cudaStreamSynchronizes
    // so this wall reflects the kernel + its sync).
    auto input = hidden_states_.index_select(0, task.token_idx).contiguous();
    c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromExternal(stream, gpu_id);
    c10::cuda::CUDAStreamGuard guard(torch_stream);
    auto output = modules_[gpu_id]->forward(input, stream);  // syncs `stream`
    auto _tc1 = std::chrono::steady_clock::now();
    compute_us_.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(_tc1 - _tw1)
            .count(),
        std::memory_order_relaxed);

    // Fail-fast #7: partial output shape must match [token_count, H].
    if (output.size(0) != task.token_idx.size(0)) {
      DLOG_FATAL("[CO_SLOT] partial shape mismatch E=", task.expert, " out_rows=",
                 output.size(0), " token_count=", task.token_idx.size(0));
    }

    // Store ONLY a partial (fp32 copy detaches from MoEMLP's shared buffer).
    Partial p;
    p.output = output.to(torch::kFloat32);
    p.token_idx = task.token_idx;
    p.weights = task.weights;
    p.layer = task.layer;
    p.expert = task.expert;
    CO_LOG("[PartialStored] E=%d token_count=%ld\n", task.expert,
           (long)task.token_idx.size(0));

    // Record this slot's compute event so a staging D2D reusing it waits.
    cudaEventRecord(slot_last_compute_event_[gpu_id][task.slot], stream);

    {
      std::lock_guard<std::mutex> lk(sched_mutex_[gpu_id]);
      p.seq = partial_seq_[gpu_id]++;
      partial_outputs_[gpu_id].push_back(std::move(p));
      slot_state_[gpu_id][task.slot] = FREE;
      // If a staging fetch is parked waiting for THIS slot, finish it now
      // (the event we just recorded gates its D2D).
      if (has_active_fetch_[gpu_id] && active_parked_[gpu_id] &&
          active_op_[gpu_id].dst_slot == task.slot) {
        CompleteStagingFetch(gpu_id);
      }
    }
    CO_LOG("[GemmDone] E=%d slot=%d\n", task.expert, task.slot);

    int prev = pending_.fetch_sub(1);
    if (prev <= 0) {
      DLOG_FATAL("[CO_SLOT] pending_ underflow (was ", prev, ")");
    }
    if (prev == 1) {  // reached 0
      std::lock_guard<std::mutex> lk(pending_mutex_);
      pending_cv_.notify_all();
    }
  }
}

// ===================================================================
// WaitLayerDone + combine_partials (revision.md §7)
// ===================================================================
torch::Tensor ExpertDispatcher::WaitLayerDone() {
  {
    std::unique_lock<std::mutex> lk(pending_mutex_);
    pending_cv_.wait(lk, [&] { return pending_.load() == 0; });
  }
  int gpu = at::cuda::current_device();
  // pending==0 ⇒ no in-flight fetch (every expert pushed + computed).
  {
    std::lock_guard<std::mutex> lk(sched_mutex_[gpu]);
    if (has_active_fetch_[gpu] || !plan_queue_[gpu].empty()) {
      DLOG_FATAL("[CO_SLOT] WaitLayerDone: pending==0 but active_fetch=",
                 (int)has_active_fetch_[gpu], " plan_queue=",
                 plan_queue_[gpu].size(), " — scheduler bug");
    }
  }
  return CombinePartials(gpu);
}

torch::Tensor ExpertDispatcher::CombinePartials(int gpu) {
  auto _tk0 = std::chrono::steady_clock::now();
  int64_t K = hidden_states_.size(0);
  int64_t H = hidden_states_.size(1);
  auto fp32_opts = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(hidden_states_.device());
  auto final_fp32 = torch::zeros({K, H}, fp32_opts);

  std::vector<Partial> parts;
  {
    std::lock_guard<std::mutex> lk(sched_mutex_[gpu]);
    parts.swap(partial_outputs_[gpu]);
  }
  // deterministic order (layer, expert, seq).  token_idx sets are disjoint
  // across experts (one expert per recv row) so order is not load-bearing for
  // correctness, only for run-to-run determinism.
  std::sort(parts.begin(), parts.end(), [](const Partial& a, const Partial& b) {
    if (a.layer != b.layer) return a.layer < b.layer;
    if (a.expert != b.expert) return a.expert < b.expert;
    return a.seq < b.seq;
  });
  int num_partials = (int)parts.size();
  for (auto& p : parts) {
    auto weighted = p.output * p.weights.unsqueeze(1);  // [K_e,H] fp32
    final_fp32.index_add_(0, p.token_idx, weighted);
  }
  auto model_dtype = hidden_states_.scalar_type();
  auto final_local = final_fp32.to(model_dtype);
  auto _tk1 = std::chrono::steady_clock::now();
  combine_us_.fetch_add(
      std::chrono::duration_cast<std::chrono::microseconds>(_tk1 - _tk0)
          .count(),
      std::memory_order_relaxed);
  CO_LOG("[LayerDone] pending=0 num_partials=%d final_shape=[%ld,%ld] "
         "dtype=%d\n",
         num_partials, (long)K, (long)H, (int)model_dtype);
  return final_local;
}

torch::Tensor ExpertDispatcher::GetPhaseTimes() {
  // [fetch_wait, weight_copy, gemm(compute), combine]
  auto options = torch::TensorOptions().dtype(torch::kInt64);
  return torch::tensor(
      {(std::int64_t)fetch_wait_us_.load(std::memory_order_relaxed),
       (std::int64_t)weight_copy_us_.load(std::memory_order_relaxed),
       (std::int64_t)compute_us_.load(std::memory_order_relaxed),
       (std::int64_t)combine_us_.load(std::memory_order_relaxed)},
      options);
}

void ExpertDispatcher::ResetPhaseTimes() {
  fetch_wait_us_.store(0, std::memory_order_relaxed);
  weight_copy_us_.store(0, std::memory_order_relaxed);
  compute_us_.store(0, std::memory_order_relaxed);
  combine_us_.store(0, std::memory_order_relaxed);
  direct_fetch_cnt_.store(0, std::memory_order_relaxed);
  staging_fetch_cnt_.store(0, std::memory_order_relaxed);
}

torch::Tensor ExpertDispatcher::GetFetchModeCounts() {
  auto options = torch::TensorOptions().dtype(torch::kInt64);
  return torch::tensor(
      {(std::int64_t)direct_fetch_cnt_.load(std::memory_order_relaxed),
       (std::int64_t)staging_fetch_cnt_.load(std::memory_order_relaxed)},
      options);
}

// ===================================================================
// introspection / stats
// ===================================================================
std::vector<std::pair<int, int>> ExpertDispatcher::GetCachedExperts(
    int gpu_id) {
  std::vector<std::pair<int, int>> result;
  if (gpu_id < 0 || gpu_id >= (int)cached_experts_.size()) return result;
  std::lock_guard<std::mutex> lock(sched_mutex_[gpu_id]);
  result.reserve(cached_experts_[gpu_id].size());
  for (auto key : cached_experts_[gpu_id]) {
    result.emplace_back((int)(key >> 32), (int)(key & 0xFFFFFFFF));
  }
  return result;
}

void ExpertDispatcher::ClearExpertCacheCounts() {
  for (auto& expert : experts_) {
    for (auto& en : expert) {
      if (en->node == nullptr) continue;
      en->node->incache_visit_count = 0;
    }
  }
}

torch::Tensor ExpertDispatcher::GetCacheStats() {
  auto options = torch::TensorOptions().dtype(torch::kInt64);
  return torch::tensor(
      {(std::int64_t)cache_visit_cnt_.load(std::memory_order_relaxed),
       (std::int64_t)cache_hit_cnt_.load(std::memory_order_relaxed),
       (std::int64_t)cache_miss_cnt_.load(std::memory_order_relaxed),
       (std::int64_t)gpu_fetch_cnt_.load(std::memory_order_relaxed),
       (std::int64_t)eviction_cnt_.load(std::memory_order_relaxed),
       (std::int64_t)overload_fetch_cnt_.load(std::memory_order_relaxed)},
      options);
}

void ExpertDispatcher::ResetCacheStats() {
  cache_visit_cnt_.store(0, std::memory_order_relaxed);
  cache_hit_cnt_.store(0, std::memory_order_relaxed);
  cache_miss_cnt_.store(0, std::memory_order_relaxed);
  gpu_fetch_cnt_.store(0, std::memory_order_relaxed);
  eviction_cnt_.store(0, std::memory_order_relaxed);
  overload_fetch_cnt_.store(0, std::memory_order_relaxed);
}
