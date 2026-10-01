// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// 2026-05-29 Controller-Owned Slot Execution (coslot) rewrite.
// =============================================================
// The dispatcher no longer chooses slots/victims or runs an autonomous LFU
// evict.  The Python GlobalCacheController computes a full FetchPlan
// (fetcher_rank / dst_slot / victim / order) per layer and hands it over via
// SubmitPlan.  Archer executes the plan deterministically:
//
//   * PlanQueue (FIFO, controller order) with at most ONE active fetch.
//   * A fetch is either a DIRECT H2D (dst_slot currently FREE) or a STAGING
//     fetch (dst_slot still COMPUTING this layer): H2D into the reserved
//     staging slot now (overlapping the prior GEMM), then — once the slot's
//     GEMM finishes and records slot_last_compute_event — a cheap D2D
//     staging->slot.  slot overwrite safety is the CUDA event.
//   * Each expert GEMM stores ONLY a partial (output + token_idx + weights).
//   * At layer end (WaitLayerDone) all partials are combined ONCE into
//     final_local [K_recv, H] (fp32 accumulate → model dtype) and returned;
//     the Python side does a single EP return all-to-all.
//
// Removed vs the old design: GPUFetchFunc thread, input_queue_, FindExpertEvict,
// gpu_overload_ busy-wait, Enqueue miss→input branch, and all the one-shot
// ExplicitFetch/Async/ReplaceAsync/FetchToStaging/PromoteStaging entry points.

#pragma once

#include <torch/extension.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/sync.h"
#include "base/noncopyable.h"
#include "base/thread.h"
#include "utils/threadsafe_queue.h"
#include "expert_module.h"

class ExpertDispatcher : public base::noncopyable {
 public:
  // ---- coslot data structures (revision.md §2.1) ----
  struct PlanOp {
    int layer = -1;
    int expert = -1;
    int dst_slot = -1;
    int victim_layer = -1;   // -1 → empty slot (no victim)
    int victim_expert = -1;
    int order = -1;          // rank-local FIFO order
  };
  struct ExecTask {
    int layer = -1;
    int expert = -1;
    int slot = -1;
    cudaEvent_t fetch_event = nullptr;   // null for hits (already resident)
    torch::Tensor token_idx;             // [K_e] long, snapshot
    torch::Tensor weights;               // [K_e] fp32, snapshot
    bool stop = false;                   // dtor sentinel
  };
  struct Partial {
    torch::Tensor output;     // [K_e, H] fp32
    torch::Tensor token_idx;  // [K_e] long
    torch::Tensor weights;    // [K_e] fp32
    int layer = -1;
    int expert = -1;
    int seq = -1;             // insertion order (deterministic combine tiebreak)
  };
  enum SlotState { FREE = 0, COMPUTING = 1 };

 public:
  explicit ExpertDispatcher(int num_experts, int num_layers, int dtype,
                            int expert_type, int num_threads = 1);
  ~ExpertDispatcher();

  // ---- per-layer inputs (recv tokens after the forward all-to-all) ----
  void SetInputs(const torch::Tensor& hidden_states,
                 const torch::Tensor& router_mask,
                 const torch::Tensor& router_weight,
                 bool is_decode = false);

  void RegisterExpert(int layer_idx, int expert_idx,
                      const std::vector<std::uint32_t>& tensor_ids,
                      std::string jit_path);

  // ---- slot pool: (cap_per_gpu + 1) slots; index cap is the staging slot. ----
  void InitSlotPool(int cap_per_gpu, int64_t expert_byte_size);
  // Explicit experiment boundary, called with a fresh empty Python controller.
  void ResetSlotPool(int cap_per_gpu);

  // ---- controller-owned execution entry points ----
  // hit_ops:  (layer, expert, slot)
  // miss_ops: (layer, expert, dst_slot, victim_layer, victim_expert, order)
  void SubmitPlan(
      int gpu,
      const std::vector<std::tuple<int, int, int>>& hit_ops,
      const std::vector<std::tuple<int, int, int, int, int, int>>& miss_ops);
  torch::Tensor WaitLayerDone();
  std::vector<std::tuple<int, torch::Tensor, torch::Tensor>> WaitLayerPartials();

  // ---- introspection / stats (kept) ----
  std::vector<std::pair<int, int>> GetCachedExperts(int gpu_id);
  std::vector<std::tuple<int, int, int>> GetCachedSlots(int gpu_id);
  void ClearExpertCacheCounts();
  torch::Tensor GetCacheStats();
  void ResetCacheStats();
  // cumulative phase wall-times [fetch_wait_us, weight_copy, gemm, combine]
  // across all exec tasks since the last ResetPhaseTimes (owner-policy bench).
  torch::Tensor GetPhaseTimes();
  void ResetPhaseTimes();
  // [direct_fetch_count, staging_fetch_count] since the last ResetPhaseTimes —
  // per-rank PCIe fetch-mode breakdown for the owner-policy fetch-workload
  // analysis (balanced balances fetch load, not routed tokens).
  torch::Tensor GetFetchModeCounts();

 private:
  void WaitForLayer();
  // scheduler (revision.md §3-4) — all under sched_mutex_[gpu].
  void StartNextFetch(int gpu);
  void DoDirectFetch(int gpu, const PlanOp& op);
  void DoStagingFetchPark(int gpu, const PlanOp& op);
  void CompleteStagingFetch(int gpu);
  void Commit(int gpu, const PlanOp& op);
  void PushExecTask(int gpu, int layer, int expert, int slot,
                    cudaEvent_t fetch_event);

  // exec worker (revision.md §5).
  void GPUExecFunc(int gpu_id);
  torch::Tensor CombinePartials(int gpu);

  // helpers
  static inline uint64_t Key(int layer, int expert) {
    return ((uint64_t)layer << 32) | (uint32_t)expert;
  }
  // snapshot of (token_idx, weights) for one expert from router_mask_ /
  // router_weight_.  token_count == 0 → FATAL (controller must filter).
  void ComputeSnapshot(int expert, torch::Tensor* token_idx,
                       torch::Tensor* weights, int64_t* token_count);

 private:
  int num_experts_;
  int dtype_;
  int expert_type_;
  std::atomic<bool> main_thread_stop_flag_;
  std::vector<std::unique_ptr<base::Thread>> threads_;

  std::vector<std::vector<ExpertNodePtr>> experts_;
  std::vector<MoEMLP*> modules_;

  // per-layer inputs
  torch::Tensor hidden_states_;   // [K_recv, H], model dtype (bf16)
  torch::Tensor router_mask_;     // [K_recv, num_experts] bool
  torch::Tensor router_weight_;   // [K_recv, num_experts] float
  std::atomic<bool> is_decode_{false};

  // slot pool (index slot_cap_ == staging)
  bool slot_pool_initialized_ = false;
  int slot_cap_ = 0;
  int64_t slot_expert_byte_size_ = 0;
  std::vector<void*> slot_pool_base_;
  std::vector<std::vector<void*>> slot_device_ptr_;             // size cap+1
  std::vector<std::vector<uint64_t>> slot_to_key_;             // size cap
  std::vector<std::unordered_map<uint64_t, int>> key_to_slot_;
  std::vector<std::unordered_set<uint64_t>> cached_experts_;
  std::vector<int64_t> cache_sizes_;                            // stats only
  std::vector<std::vector<cudaEvent_t>> slot_last_compute_event_;  // size cap+1

  // scheduler state (per gpu)
  std::vector<std::mutex> sched_mutex_;
  std::vector<std::deque<PlanOp>> plan_queue_;
  std::vector<char> has_active_fetch_;
  std::vector<char> active_parked_;
  std::vector<PlanOp> active_op_;
  std::vector<int> expected_order_;
  std::vector<std::vector<SlotState>> slot_state_;              // size cap
  std::vector<std::unordered_map<uint64_t, ExecTask>> snapshots_;  // per layer
  std::vector<std::vector<Partial>> partial_outputs_;
  std::vector<int> partial_seq_;

  // exec
  std::vector<ThreadSafeQueue<ExecTask>> exec_queue_;
  std::vector<cudaStream_t> exec_streams_;   // one per gpu (worker's stream)
  std::vector<cudaStream_t> h2d_streams_;
  std::vector<cudaEvent_t> inputs_ready_;
  std::vector<cudaEvent_t> outputs_ready_;

  // pending / layer-done sync
  std::atomic<int> pending_;
  std::mutex pending_mutex_;
  std::condition_variable pending_cv_;

  // stats
  std::atomic<std::uint64_t> cache_visit_cnt_{0};
  std::atomic<std::uint64_t> cache_hit_cnt_{0};
  std::atomic<std::uint64_t> cache_miss_cnt_{0};
  std::atomic<std::uint64_t> gpu_fetch_cnt_{0};
  std::atomic<std::uint64_t> eviction_cnt_{0};
  std::atomic<std::uint64_t> overload_fetch_cnt_{0};

  // phase wall-times (us), cumulative; reset per measured window.
  // GetPhaseTimes returns [fetch_wait, weight_copy, gemm, combine]:
  //   fetch_wait_us = cudaEventSynchronize(fetch_event) wait (host→slot H2D).
  //   weight_copy_us = SetTensorsFromIds (slot→param_ D2D weight copy).
  //   compute_us = MoEMLP.forward (the actual GEMM + its internal sync).
  //   combine_us = combine_partials.
  // (weight_copy split out so "compute" is the GEMM, not the per-expert
  //  weight movement — the two were previously conflated.)
  std::atomic<std::uint64_t> fetch_wait_us_{0};
  std::atomic<std::uint64_t> weight_copy_us_{0};
  std::atomic<std::uint64_t> compute_us_{0};
  std::atomic<std::uint64_t> combine_us_{0};
  std::atomic<std::uint64_t> direct_fetch_cnt_{0};
  std::atomic<std::uint64_t> staging_fetch_cnt_{0};
};
