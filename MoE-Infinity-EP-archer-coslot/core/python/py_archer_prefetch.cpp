// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <torch/extension.h>
#include "parallel/expert_dispatcher.h"
#include "prefetch/archer_prefetch_handle.h"
#include "model/moe.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_moe_layer", InitMoELayer,
        "Initialize the MoE layer with the specified parameters.");
  m.def("topk_softmax", TopKSoftmax,
        "Perform top-k softmax operation on the MoE layer.");

  py::class_<ArcherPrefetchHandle>(m, "prefetch_handle")
      .def(py::init<const std::string&, const double>())

      .def("offload", &ArcherPrefetchHandle::OffloadTensor)
      .def("register", (void(ArcherPrefetchHandle::*)(torch::Tensor&,
                                                      const std::uint32_t)) &
                           ArcherPrefetchHandle::RegisterTensor)
      //    .def("register",
      //         (void(ArcherPrefetchHandle::*)(torch::nn::Module&)) &
      //             ArcherPrefetchHandle::RegisterModule)
      .def("register", (void(ArcherPrefetchHandle::*)(torch::Tensor*)) &
                           ArcherPrefetchHandle::RegisterTensor)
      .def("set_tensor_device",
           (void(ArcherPrefetchHandle::*)(torch::Tensor&, torch::Device)) &
               ArcherPrefetchHandle::SetTensorDevice)
      // .def("begin", (void (ArcherPrefetchHandle::*)(torch::nn::Module&))
      // &ArcherPrefetchHandle::AcquireTensor) .def("end", (void
      // (ArcherPrefetchHandle::*)(torch::nn::Module&))
      // &ArcherPrefetchHandle::ReleaseTensor)
      .def("begin",
           (void(ArcherPrefetchHandle::*)(std::uint64_t&, torch::Tensor&)) &
               ArcherPrefetchHandle::AcquireTensor)
      .def("end",
           (void(ArcherPrefetchHandle::*)(std::uint64_t&, torch::Tensor&)) &
               ArcherPrefetchHandle::ReleaseTensor)
      // .def("begin",
      //      (void (ArcherPrefetchHandle::*)(torch::Tensor&, const
      //      std::uint32_t)) &
      //          ArcherPrefetchHandle::AcquireTensor)
      // .def("end",
      //      (void (ArcherPrefetchHandle::*)(torch::Tensor&, const
      //      std::uint32_t)) &
      //          ArcherPrefetchHandle::ReleaseTensor)
      //    .def("get_trace",
      //    (torch::Tensor(ArcherPrefetchHandle::*)()) &
      //    ArcherPrefetchHandle::GetTrace)
      .def("get_hit_rate", (torch::Tensor(ArcherPrefetchHandle::*)()) &
                               ArcherPrefetchHandle::GetHitRate)
      .def("set_trace", (void(ArcherPrefetchHandle::*)(const torch::Tensor&)) &
                            ArcherPrefetchHandle::SetTrace)
      //    .def("trace_request",
      //         (void(ArcherPrefetchHandle::*)(const std::uint64_t, const
      //         std::uint32_t)) &
      //             ArcherPrefetchHandle::TraceRequest)
      .def("set_topology",
           (void(ArcherPrefetchHandle::*)(
               const std::vector<std::tuple<
                   std::string, std::vector<std::vector<TensorID>>>>&)) &
               ArcherPrefetchHandle::SetTopology)
      .def("update_tensor_map",
           (void(ArcherPrefetchHandle::*)(std::uint64_t, std::uint64_t)) &
               ArcherPrefetchHandle::UpdateTensorMap)
      .def("is_tensor_offloaded", &ArcherPrefetchHandle::IsTensorOffloaded)
      .def("is_tensor_index_initialized",
           &ArcherPrefetchHandle::IsTensorIndexInitialized)
      .def("is_tensor_on_device",
           (bool(ArcherPrefetchHandle::*)(const torch::Tensor&) const) &
               ArcherPrefetchHandle::IsTensorOnDevice)
      .def("is_tensor_on_device",
           (bool(ArcherPrefetchHandle::*)(const std::uint32_t) const) &
               ArcherPrefetchHandle::IsTensorOnDevice)
      .def("get_node_default_device",
           &ArcherPrefetchHandle::GetNodeDefaultDevice)
      .def("get_node_device", &ArcherPrefetchHandle::GetNodeDevice)
      .def("prefetch_tensors", &ArcherPrefetchHandle::PrefetchTensors)
      .def("replace_cache_candidates",
           &ArcherPrefetchHandle::ReplaceCacheCandidates)
      .def("enqueue_prefetch", &ArcherPrefetchHandle::EnqueuePrefetch)
      .def("fetch_tensors", &ArcherPrefetchHandle::FetchTensors)
      .def("clean_up_resources", &ArcherPrefetchHandle::CleanUpResources);
  //    .def("set_node_cache_priority",
  //    &ArcherPrefetchHandle::SetNodeCachePriority);

  // 2026-05-29 Controller-Owned Slot Execution (coslot).  The dispatcher takes
  // a full per-layer FetchPlan from the Python GlobalCacheController and
  // executes it deterministically; it no longer picks slots/victims or runs an
  // autonomous evict.  The old per-op enqueue / explicit_* / staging entry
  // points are gone — submit_plan + wait_layer_done replace them.
  py::class_<ExpertDispatcher>(m, "expert_dispatcher")
      .def(py::init<int, int, int, int, int>())
      .def("register_expert", &ExpertDispatcher::RegisterExpert)
      .def("set_inputs", &ExpertDispatcher::SetInputs,
           py::arg("hidden_states"), py::arg("router_mask"),
           py::arg("router_weight"), py::arg("is_decode") = false,
           "Set this layer's recv tokens (hidden / one-hot router_mask / "
           "router_weight) before submit_plan.")
      .def("init_slot_pool", &ExpertDispatcher::InitSlotPool,
           py::arg("cap_per_gpu"), py::arg("expert_byte_size"),
           "Pre-allocate (cap_per_gpu + 1) fixed slots of expert_byte_size "
           "each per GPU (slot index cap_per_gpu is the reserved staging "
           "slot).  Call ONCE after model load.  expert_byte_size<=0 → "
           "auto-detect the max registered expert size.")
      .def("submit_plan", &ExpertDispatcher::SubmitPlan,
           py::arg("gpu"), py::arg("hit_ops"), py::arg("miss_ops"),
           "Submit this layer's FetchPlan.  hit_ops: list[(layer, expert, "
           "slot)] already resident.  miss_ops: list[(layer, expert, "
           "dst_slot, victim_layer, victim_expert, order)] to fetch in the "
           "given rank-local order.  Kicks the PlanQueue (one active fetch, "
           "direct or staging) and the exec workers.")
      .def("wait_layer_done", &ExpertDispatcher::WaitLayerDone,
           "Block until every submitted expert's GEMM is done, combine all "
           "partials once into final_local [K_recv, H] (fp32 accumulate → "
           "model dtype) and return it.")
      .def("get_cached_experts", &ExpertDispatcher::GetCachedExperts,
           py::arg("gpu_id"),
           "Snapshot of currently cached (layer, expert) pairs on a GPU.")
      .def("clear_expert_cache_counts",
           &ExpertDispatcher::ClearExpertCacheCounts)
      .def("get_cache_stats", &ExpertDispatcher::GetCacheStats)
      .def("reset_cache_stats", &ExpertDispatcher::ResetCacheStats)
      .def("get_phase_times", &ExpertDispatcher::GetPhaseTimes,
           "Cumulative [fetch_wait_us, compute_us, combine_us] across exec "
           "tasks since reset_phase_times.")
      .def("reset_phase_times", &ExpertDispatcher::ResetPhaseTimes)
      .def("get_fetch_mode_counts", &ExpertDispatcher::GetFetchModeCounts,
           "[direct_fetch_count, staging_fetch_count] since reset_phase_times.");
}
