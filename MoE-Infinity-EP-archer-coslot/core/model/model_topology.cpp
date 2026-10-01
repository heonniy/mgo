// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "model_topology.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include "aio/archer_prio_aio_handle.h"
#include "aio/archer_tensor_handle.h"
#include "aio/archer_tensor_index.h"
#include "common/time.h"
#include "common/types.h"
#include "memory/memory_pool.h"
#include "memory/stream_pool.h"
#include "parallel/expert_dispatcher.h"
#include "prefetch/task_scheduler.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "utils/tqdm.h"

// cudaStream_t kCudaStreamH2D = NULL;
std::unique_ptr<ArcherTopologyHandle> kTopologyHandle = nullptr;

const std::string Node::str() {
  // write same string using c style sprintf
  std::stringstream ss;
  for (auto& tensor_id : tensor_ids) {
    ss << tensor_id << ",";
  }

  char buffer[1024];
  memset(buffer, 0, 1024);
  sprintf(buffer, "ID[%ld,%lx] (%ldMB) STATE(%d) TENSOR[%s] DEVICE[%s;%s;%s];",
          id, corr_id, byte_size / MB, state.load(), ss.str().c_str(),
          device.str().c_str(), default_device.str().c_str(),
          default_host.str().c_str());

  return std::string(buffer);
}

Node::Node()
    : corr_id(0),
      byte_size(0),
      last_access_time(MCIROSECONDS_SINCE_EPOCH),
      device(DISK_DEVICE),
      default_device(DEFAULT_CUDA_DEVICE) {}

Node::~Node() {
  // 2026-05-27: clean up fetch_event if it was created by SetDeviceFromHostAsync.
  // cudaEventDestroy is safe even on a NULL event; we still null-check to
  // avoid a no-op CUDA call during process teardown when the runtime might
  // be partly torn down already.
  if (fetch_event != nullptr) {
    cudaEventDestroy(fetch_event);
    fetch_event = nullptr;
  }
  // 2026-05-27 cooperative-offloading: compute_event cleanup (I5).
  if (compute_event != nullptr) {
    cudaEventDestroy(compute_event);
    compute_event = nullptr;
  }
}

// 2026-05-27 cooperative-offloading I5: GEMM 끝에서 compute_event 를
// exec_stream 에 record.  ExplicitReplaceAsync 가 victim 의 이 event 를
// h2d_stream 에 wait → fetch-write 가 compute-read 이후 GPU side 에서만
// 진행.  CPU block 없음.
//
// Lazy create — kernel 한 번도 실행 안 한 Node 에는 event 안 만듦.
void Node::RecordComputeEvent(cudaStream_t exec_stream) {
  if (compute_event == nullptr) {
    auto err =
        cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      DLOG_FATAL("RecordComputeEvent: cudaEventCreate failed: ",
                 cudaGetErrorString(err));
    }
  }
  auto err = cudaEventRecord(compute_event, exec_stream);
  if (err != cudaSuccess) {
    DLOG_FATAL("RecordComputeEvent: cudaEventRecord failed: ",
               cudaGetErrorString(err));
  }
}

// 2026-05-27: async host->GPU transition for streaming fetch.
//
// SAFETY MODEL:
//   1. Caller (ExplicitFetchAsync) holds this->mutex throughout the call.
//   2. Device allocation is synchronous (cudaMalloc returns when memory is
//      reserved) so device_memory_ptr is valid by function return.
//   3. CudaMemcpyAsync queues the copy on h2d_stream — does NOT block.
//   4. SetModuleCudaMemoryFromCPU does only pointer-aliasing for torch
//      tensor views — safe to call while H2D is still in flight.
//   5. cudaEventRecord(fetch_event, h2d_stream) records "the H2D queued
//      above is complete at this point".  Consumers cudaStreamWaitEvent or
//      cudaEventSynchronize on this to gate their reads / frees.
//   6. `device = target_device` is set LAST so that any other thread checking
//      is_cuda() while we are mid-setup still sees a coherent (pre or post)
//      state — never a half-initialized one.  The mutex makes this strict.
//
// CONSUMERS:
//   * GPUExecFunc: cudaStreamWaitEvent(exec_stream, fetch_event, 0) before
//     launching the SwiGLU kernel — kernel waits for the H2D on the GPU.
//   * SetDevice GPU->CPU branch (eviction): cudaEventSynchronize(fetch_event)
//     before freeing device_memory_ptr — CPU waits so the free is safe.
void Node::SetDeviceFromHostAsync(const torch::Device& target_device,
                                  cudaStream_t h2d_stream) {
  // Pre-conditions enforced by caller (ExplicitFetchAsync), checked here
  // for defense-in-depth.  These are programming errors, not runtime
  // failures — DLOG_FATAL is correct.
  if (!device.is_cpu()) {
    DLOG_FATAL("SetDeviceFromHostAsync: src device must be CPU, got ",
               device.str());
  }
  if (!target_device.is_cuda()) {
    DLOG_FATAL("SetDeviceFromHostAsync: target must be CUDA, got ",
               target_device.str());
  }
  if (host_memory_ptr == nullptr) {
    DLOG_FATAL("SetDeviceFromHostAsync: host_memory_ptr is null (id=", id,
               ").  Sparse experts must have persistent host backing.");
  }
  if (device_memory_ptr != nullptr) {
    DLOG_FATAL("SetDeviceFromHostAsync: device_memory_ptr already allocated "
               "(id=", id, ").  Double-fetch without intervening evict.");
  }

  // 1. Synchronous device allocation.
  device_memory_ptr =
      kDeviceMemoryPool->AllocateMemory(id, byte_size, target_device);
  assert(device_memory_ptr != nullptr);

  // 2. Async H2D copy on the dedicated stream.  No synchronize.
  CudaMemcpyAsync(device_memory_ptr, host_memory_ptr, byte_size,
                  cudaMemcpyHostToDevice, h2d_stream);

  // 3. Update torch tensor views to point at the device buffer.  This is
  // pointer aliasing only — does NOT read device data.  Safe to run while
  // the H2D copy is still streaming.
  SetModuleCudaMemoryFromCPU(tensor_ids, device_memory_ptr, target_device);

  // 4. Record event after the H2D is queued.  fetch_event is lazily created
  // per node on first async fetch; reused on subsequent fetches (re-record
  // overwrites the prior signal point).
  if (fetch_event == nullptr) {
    // Make sure the event is created on the right device.
    cudaSetDevice(target_device.index());
    auto err = cudaEventCreateWithFlags(&fetch_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      DLOG_FATAL("SetDeviceFromHostAsync: cudaEventCreate failed: ",
                 cudaGetErrorString(err));
    }
  }
  auto err = cudaEventRecord(fetch_event, h2d_stream);
  if (err != cudaSuccess) {
    DLOG_FATAL("SetDeviceFromHostAsync: cudaEventRecord failed: ",
               cudaGetErrorString(err));
  }

  // 5. Mark device flag LAST.  Under this->mutex, so observers either see
  // is_cpu()==true (pre-fetch) or is_cuda()==true with fetch_event recorded
  // (post-fetch) — never an intermediate state.
  device = target_device;
}

void Node::SetDevice(const torch::Device& target_device, bool on_demand,
                     cudaStream_t stream) {
  DLOG_TRACE("SetDevice: " + str() + " to " + target_device.str());
  if (device == target_device) {
    DLOG_TRACE("SetDevice: " + str() + " to " + target_device.str() +
               " but device is the same");
    return;
  }

  if (device.type() == target_device.type()) {
    DLOG_WARN("SetDevice: " + str() + " to " + target_device.str() +
              " but device type is the same");
    return;
  }

  // if (kCudaStreamH2D == NULL) {
  //     auto cudaError = cudaStreamCreateWithFlags(&kCudaStreamH2D,
  //     cudaStreamNonBlocking); if (cudaError != cudaSuccess) {
  //         DLOG_ERROR("cudaStreamCreate failed: {}",
  //         cudaGetErrorString(cudaError)); exit(-1);
  //     }
  // }

  if (target_device == DISK_DEVICE) {
    SetModuleDisk(tensor_ids);
    if (host_memory_ptr != nullptr) {
      kHostMemoryPool->FreeMemory(id, host_memory_ptr, byte_size, CPU_DEVICE);
      host_memory_ptr = nullptr;
    }
    if (device_memory_ptr != nullptr) {
      kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
      device_memory_ptr = nullptr;
    }
  } else {
    // both are null, which means the node is not initialized
    bool from_disk =
        (host_memory_ptr == nullptr && device_memory_ptr == nullptr);

    if (from_disk && target_device.is_cuda()) {
      // Pipelined path: disk -> host -> GPU with per-tensor overlap
      host_memory_ptr =
          kHostMemoryPool->AllocateMemory(id, byte_size, CPU_DEVICE);
      assert(host_memory_ptr != nullptr);
      device_memory_ptr =
          kDeviceMemoryPool->AllocateMemory(id, byte_size, target_device);
      assert(device_memory_ptr != nullptr);

      cudaStream_t h2d_stream = stream;
      bool own_stream = false;
      if (h2d_stream == nullptr) {
        cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking);
        own_stream = true;
      }

      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      std::int64_t param_offset = 0;
      for (const auto& tensor_id : tensor_ids) {
        // Read tensor from disk into host buffer
        kArcherTensorHandle->ReadTensor(
            tensor_id, static_cast<char*>(host_memory_ptr) + param_offset,
            on_demand);

        auto it = kTensorIndex->find(tensor_id);
        std::int64_t size_aligned =
            (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);

        // Async copy this tensor's data to GPU (overlaps with next disk read)
        CudaMemcpyAsync(static_cast<char*>(device_memory_ptr) + param_offset,
                        static_cast<char*>(host_memory_ptr) + param_offset,
                        size_aligned, cudaMemcpyHostToDevice, h2d_stream);

        param_offset += size_aligned;
      }
      cudaStreamSynchronize(h2d_stream);
      if (own_stream) {
        cudaStreamDestroy(h2d_stream);
      }

      // Create torch tensor views on both host and device buffers.
      // SetModuleCudaMemoryFromCPU overwrites it->second.tensor (set by
      // SetModuleMemoryFromDisk_Views) with the device view — after this
      // line no torch tensor references host_memory_ptr.
      SetModuleMemoryFromDisk_Views(tensor_ids, host_memory_ptr);
      SetModuleCudaMemoryFromCPU(tensor_ids, device_memory_ptr, target_device);
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      DLOG_TRACE("PipelinedDiskToGpu time: {} us", end_time - start_time);

      // Dense host-pinned buffer leak fix (2026-05-27): the Pipelined path
      // is reached ONLY for dense nodes (init's dense loop calls SetDevice
      // with target=CUDA; sparse goes disk->CPU first via the else-branch
      // below).  Dense is never evicted (FindExpertEvict + RemoveCachedSparse
      // are sparse-only; autonomous archer evict disabled by default).
      // Keeping host_memory_ptr alive permanently wastes (dense_bytes ×
      // ranks-per-NUMA) pinned host RAM per NUMA — measurable cgroup
      // pressure with 4+ ranks.  Free now; the GPU->CPU eviction path
      // re-allocates defensively in case any future code path moves dense
      // off-GPU.
      if (!is_sparse && host_memory_ptr != nullptr) {
        kHostMemoryPool->FreeMemory(id, host_memory_ptr, byte_size, CPU_DEVICE);
        host_memory_ptr = nullptr;
      }
    } else if (from_disk) {
      // CPU-only target: use original sequential path.
      // MOE_ARCHER_DBG_SPARSE=1 surfaces step-by-step latency so we can tell
      // whether a stuck rank is hanging in AllocateMemory (host-pool mutex)
      // or in SetModuleMemoryFromDisk (archer ReadTensor / disk I/O).
      const bool _dbg = std::getenv("MOE_ARCHER_DBG_SPARSE") != nullptr &&
                        std::string(std::getenv("MOE_ARCHER_DBG_SPARSE")) == "1";
      const char* _env_rank2 = std::getenv("RANK");
      const std::string _rk = _env_rank2 ? _env_rank2 : "?";

      std::int64_t _t_alloc0 = _dbg ? MCIROSECONDS_SINCE_EPOCH : 0;
      host_memory_ptr =
          kHostMemoryPool->AllocateMemory(id, byte_size, CPU_DEVICE);
      assert(host_memory_ptr != nullptr);
      if (_dbg) {
        std::int64_t _t_alloc1 = MCIROSECONDS_SINCE_EPOCH;
        std::cerr << "[SETDEV rank=" << _rk << " id=" << id
                  << " size=" << byte_size
                  << "] alloc=" << (_t_alloc1 - _t_alloc0) << "us" << std::endl;
        std::cerr.flush();
      }

      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      SetModuleMemoryFromDisk(tensor_ids, host_memory_ptr, on_demand);
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      DLOG_TRACE("SetModuleMemoryFromDisk time: {} us", end_time - start_time);
      if (_dbg) {
        std::cerr << "[SETDEV rank=" << _rk << " id=" << id
                  << "] read=" << (end_time - start_time) << "us" << std::endl;
        std::cerr.flush();
      }
    }

    if (!from_disk && target_device.is_cuda()) {
      // Already in host memory, just copy to GPU
      device_memory_ptr =
          kDeviceMemoryPool->AllocateMemory(id, byte_size, target_device);
      assert(device_memory_ptr != nullptr);
      assert(host_memory_ptr != nullptr);

      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      if (stream == nullptr) {
        CudaMemcpy(device_memory_ptr, host_memory_ptr, byte_size,
                   cudaMemcpyHostToDevice);
      } else {
        CudaMemcpyAsync(device_memory_ptr, host_memory_ptr, byte_size,
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
      }
      SetModuleCudaMemoryFromCPU(tensor_ids, device_memory_ptr, target_device);
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      DLOG_TRACE("SetModuleCudaMemoryFromCPU time: {} us",
                 end_time - start_time);
    }

    if (target_device.is_cpu() && device.is_cuda()) {
      // host_memory_ptr may be null for dense nodes whose host staging
      // buffer was released after the Pipelined disk->GPU load (see leak
      // fix above).  Dense is never expected to take this branch under our
      // production config, but if any future code path moves dense off-GPU
      // we re-allocate the host destination here so the device->host copy
      // has somewhere to land — instead of crashing on a stale assertion.
      if (host_memory_ptr == nullptr) {
        host_memory_ptr =
            kHostMemoryPool->AllocateMemory(id, byte_size, CPU_DEVICE);
        assert(host_memory_ptr != nullptr);
      }
      // 2026-05-27 streaming-fetch safety: if there is an in-flight async
      // H2D for this node (recorded by SetDeviceFromHostAsync), CPU-side
      // synchronize on its event before freeing device_memory_ptr.  Without
      // this, the cudaFree below could run while the H2D copy is still
      // touching the memory — undefined behavior.
      //
      // In normal flow, GPUExecFunc has already cudaStreamWaitEvent'd and
      // the kernel has completed before this eviction is called, so the
      // event is signaled and Synchronize returns immediately.  This is
      // defense-in-depth for cross-layer races (evict at layer N+1 on an
      // async-fetched expert from layer N whose kernel happened to be
      // skipped — should never happen, but cheap to guard).
      if (fetch_event != nullptr) {
        cudaEventSynchronize(fetch_event);
      }
      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      SetModuleMemoryFromCuda(tensor_ids, host_memory_ptr);
      kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
      device_memory_ptr = nullptr;
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      DLOG_TRACE("SetModuleMemoryFromCuda time: {} us", end_time - start_time);
    }
  }
  device = target_device;
}

ArcherTopologyHandle::ArcherTopologyHandle() {}

NodePtrList ArcherTopologyHandle::GetLFUNodes(const torch::Device& device) {
  NodePtrList nodes;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto node_body : lfu_nodes_) {
    CONTINUE_IF_NULL(node_body);
    if (node_body->node->device == device) {
      nodes.push_back(node_body->node);
    }
  }
  return nodes;
}

NodePtrList ArcherTopologyHandle::GetDenseNodes() {
  NodePtrList nodes;
  for (auto stage : pipeline_.stages) {
    if (stage->is_sparse) {
      continue;
    }
    for (auto node_body : stage->nodes) {
      nodes.push_back(node_body->node);
    }
  }
  return nodes;
}
NodePtrList ArcherTopologyHandle::GetSparseNodes() {
  NodePtrList nodes;
  for (auto stage : pipeline_.stages) {
    if (!stage->is_sparse) {
      continue;
    }
    for (auto node_body : stage->nodes) {
      nodes.push_back(node_body->node);
    }
  }
  return nodes;
}

NodePtrList ArcherTopologyHandle::GetDenseNodes(const NodePtr& node,
                                                const std::size_t& k) {
  NodePtrList nodes;

  std::size_t low_corr_id = node->corr_id & 0xFFFFFFFF;  // stage id
  std::size_t high_corr_id = node->corr_id >> 32;        // node id
  bool is_last_node = (0xFFFFFFFF == high_corr_id);
  if (is_last_node) {
    high_corr_id = 0;  // reset to 0 avoid miss use
  }

  std::lock_guard<std::mutex> lock(mutex_);

  low_corr_id++;
  std::size_t count = 0;
  while ((low_corr_id < pipeline_.stages.size()) && (count < k)) {
    // Due to MoE design, we only process layer by layer
    auto stage = pipeline_.stages[low_corr_id];
    low_corr_id++;
    if (stage->is_sparse) {
      continue;
    }

    nodes.push_back(stage->nodes[0]->node);
    count++;
  }
  return nodes;
}

NodePtrList ArcherTopologyHandle::GetSparseNodes(const NodePtr& node,
                                                 const std::size_t& k) {
  NodePtrList nodes;

  std::size_t low_corr_id = node->corr_id & 0xFFFFFFFF;  // stage id
  std::size_t high_corr_id = node->corr_id >> 32;        // node id
  bool is_last_node = (0xFFFFFFFF == high_corr_id);
  if (is_last_node) {
    high_corr_id = 0;  // reset to 0 avoid miss use
  }

  std::lock_guard<std::mutex> lock(mutex_);

  low_corr_id++;
  std::size_t count = 0;
  while ((low_corr_id < pipeline_.stages.size()) && (count < k)) {
    // Due to MoE design, we only process layer by layer
    auto stage = pipeline_.stages[low_corr_id];

    low_corr_id++;
    if (!stage->is_sparse) {
      continue;
    }

    nodes.push_back(stage->nodes[0]->node);
    count++;
  }
  return nodes;
}

std::uint64_t ArcherTopologyHandle::GetLastActivateStage(
    const HashID& hash_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = last_active_stage_.find(hash_id);
  if (it == last_active_stage_.end()) {
    return 0;
  }
  return it->second;
}

std::vector<std::vector<std::size_t>>
ArcherTopologyHandle::GetNodeVisitCounts() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::vector<std::size_t>> node_visit_counts;
  for (auto& stage : pipeline_.stages) {
    for (auto& node : stage->nodes) {
      node->node->io_state = NODE_STATE_NONE;
      std::vector<std::size_t> metrics{node->visit_cnt,
                                       node->gpu_visit_cnt,
                                       node->cpu_visit_cnt,
                                       node->hit_cnt,
                                       node->gpu_hit_cnt,
                                       node->cpu_hit_cnt,
                                       node->node->tensor_ids.size(),
                                       node->prefetch_cnt,
                                       node->node->unused_count,
                                       node->node->io_state,
                                       node->is_sparse};
      node_visit_counts.push_back(metrics);
    }
  }
  return node_visit_counts;
}

std::vector<std::size_t> ArcherTopologyHandle::GetChildVisitCounts() {
  std::lock_guard<std::mutex> lock(mutex_);
  int num_layers = 0;
  int num_experts = 0;
  for (auto& stage : pipeline_.stages) {
    if (stage->is_sparse) {
      num_layers += 1;
      num_experts = stage->nodes.size();
    }
  }
  std::vector<std::size_t> child_visit_counts((num_layers - 1) * num_experts *
                                              num_experts);
  int layer_idx = 0;
  int parent_idx = 0;
  int expert_idx = 0;
  for (auto& stage : pipeline_.stages) {
    if (stage->is_sparse) {
      for (auto& node : stage->nodes) {
        if (node->children.size() > 0) {
          for (auto& count : node->children_visit_cnt) {
            child_visit_counts[layer_idx * num_experts * num_experts +
                               parent_idx * num_experts + expert_idx] = count;
            expert_idx++;
          }
        }
        parent_idx++;
        expert_idx = 0;
      }
      layer_idx++;
      parent_idx = 0;
    }
  }

  return child_visit_counts;
}

void ArcherTopologyHandle::SetNodeVisitCounts(
    const std::vector<std::size_t>& visit_counts) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t num_nodes = 0;
  std::size_t num_experts = 0;
  for (auto& stage : pipeline_.stages) {
    if (stage->is_sparse) {
      num_nodes += stage->nodes.size();
      num_experts = stage->nodes.size();
    }
  }
  if (visit_counts.size() != num_nodes) {
    DLOG_ERROR("visit_counts size {} not equal to num_nodes {}",
               visit_counts.size(), num_nodes);
    return;
  }

  int layer_idx = 0;
  int expert_idx = 0;
  for (auto& stage : pipeline_.stages) {
    if (stage->is_sparse) {
      for (auto& node : stage->nodes) {
        node->visit_cnt = visit_counts[layer_idx * num_experts + expert_idx];
        expert_idx++;
      }
      layer_idx++;
      expert_idx = 0;
    }
  }

  DisableTrace();
}
void ArcherTopologyHandle::SetChildVisitCounts(
    const std::vector<std::size_t>& visit_counts) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t num_layers = 0;
  std::size_t num_experts = 0;
  for (auto& stage : pipeline_.stages) {
    if (stage->is_sparse) {
      num_layers += 1;
      num_experts = stage->nodes.size();
    }
  }
  if (visit_counts.size() != (num_layers - 1) * num_experts * num_experts) {
    DLOG_ERROR("visit_counts size {} not equal to num_layers {}",
               visit_counts.size(), num_layers);
    return;
  }

  int layer_idx = 0;
  int parent_idx = 0;
  int expert_idx = 0;
  for (auto& stage : pipeline_.stages) {
    if (stage->is_sparse) {
      for (auto& node : stage->nodes) {
        if (node->children.size() > 0) {
          for (auto& count : node->children_visit_cnt) {
            count = visit_counts[layer_idx * num_experts * num_experts +
                                 parent_idx * num_experts + expert_idx];
            expert_idx++;
          }
        }
        parent_idx++;
        expert_idx = 0;
      }
      layer_idx++;
      parent_idx = 0;
    }
  }

  DisableTrace();
}

bool ArcherTopologyHandle::IsLastNode(const NodePtr& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto last_stage_ptr = pipeline_.stages.back();
  auto& nodes = last_stage_ptr->nodes;
  for (auto& n : nodes) {
    if (n->node == node) {
      return true;
    }
  }
  return false;
}
bool ArcherTopologyHandle::IsFirstNode(const NodePtr& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto first_stage_ptr = pipeline_.stages.front();
  auto& nodes = first_stage_ptr->nodes;
  for (auto& n : nodes) {
    if (n->node == node) {
      return true;
    }
  }
  return false;
}

void ArcherTopologyHandle::InitializeTopology(
    const std::vector<
        std::tuple<std::string, std::vector<std::vector<TensorID>>>>&
        topology) {
  std::lock_guard<std::mutex> lock(mutex_);
  pipeline_.stages.clear();
  std::size_t node_id = 0;
  std::size_t layer_id = 0;
  std::size_t last_sparse_layer_id = UINT64_MAX;

  size_t num_sparse_layers = 0;
  size_t num_experts = 0;

  std::vector<NodePtr> all_nodes;

  for (auto& stage : topology) {
    auto& stage_tensors = std::get<1>(stage);
    auto stage_ptr = std::make_shared<Stage>(stage_tensors.size() > 1);

    std::size_t expert_id = 0;
    for (auto& tensor_ids : stage_tensors) {
      auto node_ptr = std::make_shared<Node>();
      node_ptr->tensor_ids = tensor_ids;
      int64_t byte_size = 0;
      for (auto& tensor_id : tensor_ids) {
        auto it = kTensorIndex->find(tensor_id);
        if (it != kTensorIndex->end()) {
          std::int64_t size_aligned =
              (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
          byte_size += size_aligned;
        } else {
          DLOG_ERROR("Tensor {} not found in tensor index", tensor_id);
        }
      }
      node_ptr->byte_size = byte_size;
      node_ptr->id = node_id;
      node_ptr->corr_id =
          (layer_id & 0xFFFFFFFF) | ((expert_id & 0xFFFFFFFF) << 32);
      node_ptr->is_sparse = stage_ptr->is_sparse;

      all_nodes.push_back(node_ptr);

      auto node_body_ptr = std::make_shared<NodeBody>(node_ptr);
      node_body_ptr->is_sparse = stage_ptr->is_sparse;

      stage_ptr->nodes.push_back(node_body_ptr);

      node_id++;
      expert_id++;
    }
    pipeline_.stages.push_back(stage_ptr);
    auto current_layer_id = layer_id;
    layer_id++;

    if (stage_ptr->is_sparse) {
      if (UINT64_MAX == last_sparse_layer_id) {
        last_sparse_layer_id = current_layer_id;
        continue;
      }
      // set node_body_ptr vectors to be the same size as the number of experts
      // all counts initialized to 0
      auto last_sparse_stage_ptr = pipeline_.stages[last_sparse_layer_id];
      for (auto& node : last_sparse_stage_ptr->nodes) {
        node->children_visit_cnt.resize(stage_ptr->nodes.size(), 0);
        node->children = stage_ptr->nodes;
      }
      last_sparse_layer_id = current_layer_id;

      num_sparse_layers++;
      num_experts = stage_ptr->nodes.size();
    }
  }

  // set last stage nodes corr_id higher 32 bits to be 0xFFFFFFFF
  auto last_stage_ptr = pipeline_.stages.back();
  for (auto& node_body : last_stage_ptr->nodes) {
    node_body->node->corr_id =
        (node_body->node->corr_id & 0xFFFFFFFF) | (UINT64_MAX << 32);
  }

  // output every tensor id in node
  for (auto& stage : pipeline_.stages) {
    for (auto& node : stage->nodes) {
      std::stringstream ss;
      for (auto& tensor_id : node->node->tensor_ids) {
        ss << tensor_id << " ";
      }
      // DLOG_TRACE("Node {} tensor ids {}", node->node->id, ss.str());
      lfu_nodes_.push_back(node);
    }
  }

  DLOG_TRACE("InitializeTopology pipeline_.stages.size() {}",
             pipeline_.stages.size());

  // Model placement
  auto num_gpu = GetDeviceCount();
  std::vector<std::int64_t> free_device_mem(num_gpu, 0);
  for (int i = 0; i < num_gpu; i++) {
    free_device_mem[i] =
        kDeviceMemoryPool->GetMemoryCapacity(torch::Device(torch::kCUDA, i));
  }

  auto sparse_nodes = GetSparseNodes();
  auto dense_nodes = GetDenseNodes();

  DLOG_TRACE(
      "InitializeTopology num_gpu {} sparse_nodes.size() {} dense_nodes.size() "
      "{}",
      num_gpu, sparse_nodes.size(), dense_nodes.size());

  int target_device_id = 0;
  // int dense_gpu_idx = 0;
  // int sparse_gpu_idx = 0;

  // Split evently dense nodes only
  int num_dense_nodes_per_device = std::max(
      1, static_cast<int>(
             std::ceil(static_cast<double>(dense_nodes.size()) / num_gpu / 2)));
  // int total_dense_nodes = dense_nodes.size();
  int counter = 0;
  DLOG_INFO("Moving dense parameters to GPU");
  auto last_dense_node = dense_nodes.empty() ? nullptr : dense_nodes.back();
  for (auto& node_ptr : tqdm::tqdm(dense_nodes)) {
    int node_device_id =
        node_ptr == last_dense_node ? num_gpu - 1 : target_device_id;
    node_ptr->default_device = torch::Device(torch::kCUDA, node_device_id);
    counter++;
    if (counter % num_dense_nodes_per_device == 0) {
      target_device_id = (target_device_id + 1) % num_gpu;
    }
    node_ptr->SetDevice(node_ptr->default_device, false);
  }

  // NUMA-shared host memory: when MOE_INFINITY_SHM_BASE_PTR is set, the
  // sparse-load HostMemoryPool allocations go into the shared region.
  // Leader/follower processes hit AllocateMemory in the same deterministic
  // order, so offsets match. Leader's SetModuleMemoryFromDisk fills the
  // bytes; follower (MOE_INFINITY_NUMA_FOLLOWER=1) skips the disk read.
  extern void _set_use_shm_alloc(bool);
  _set_use_shm_alloc(true);
  DLOG_INFO("Moving sparse parameters to CPU (sparse_nodes=",
            sparse_nodes.size(), ")");

  // Diagnostic instrumentation (env-gated: MOE_ARCHER_DBG_SPARSE=1) to
  // locate which sparse_node's SetDevice() hangs in the random-race case
  // (rank-1 stuck in v10 dbg, rank-2 in v4 — different ranks each run).
  // Logs first 3, last 50, and every 1000th node's BEGIN/END+latency on
  // stderr with the process's RANK env var so 4-rank interleaved output
  // is still attributable.
  const bool _dbg_sparse =
      std::getenv("MOE_ARCHER_DBG_SPARSE") != nullptr &&
      std::string(std::getenv("MOE_ARCHER_DBG_SPARSE")) == "1";
  const char* _env_rank = std::getenv("RANK");
  const std::string _dbg_rank = _env_rank ? _env_rank : "?";
  const int _dbg_total = static_cast<int>(sparse_nodes.size());
  int _dbg_idx = 0;
  for (auto& node_ptr : tqdm::tqdm(sparse_nodes)) {
    const bool _log =
        _dbg_sparse &&
        (_dbg_idx < 3 || (_dbg_total - _dbg_idx) <= 50 ||
         _dbg_idx % 1000 == 0);
    std::int64_t _t0 = 0;
    if (_log) {
      _t0 = MCIROSECONDS_SINCE_EPOCH;
      std::cerr << "[SPARSE rank=" << _dbg_rank << " idx=" << _dbg_idx
                << "/" << _dbg_total
                << " node_id=" << node_ptr->id << "] BEGIN" << std::endl;
      std::cerr.flush();
    }
    node_ptr->default_device = torch::Device(torch::kCUDA, target_device_id);
    target_device_id = (target_device_id + 1) % num_gpu;
    node_ptr->SetDevice(CPU_DEVICE, false);
    if (_log) {
      std::int64_t _t1 = MCIROSECONDS_SINCE_EPOCH;
      std::cerr << "[SPARSE rank=" << _dbg_rank << " idx=" << _dbg_idx
                << "/" << _dbg_total
                << " node_id=" << node_ptr->id << "] END "
                << (_t1 - _t0) << "us" << std::endl;
      std::cerr.flush();
    }
    _dbg_idx++;
  }
  _set_use_shm_alloc(false);
  if (_dbg_sparse) {
    std::cerr << "[SPARSE rank=" << _dbg_rank << "] LOOP_EXIT "
              << "(processed " << _dbg_idx << "/" << _dbg_total << ")"
              << std::endl;
    std::cerr.flush();
  }

  DLOG_TRACE("InitializeTopology pipeline_.stages.size() {}",
             pipeline_.stages.size());

  for (auto& node_ptr : all_nodes) {
    DLOG_TRACE("Node {} {} device {}", node_ptr->id, node_ptr->is_sparse,
               node_ptr->default_device.str());
  }

  EnableTrace();
}

NodePtr ArcherTopologyHandle::GetNodeFromTensorID(const TensorID& tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = tensor_id_to_node_.find(tensor_id);
  if (it != tensor_id_to_node_.end()) {
    return it->second;
  } else {
    // search in pipeline
    for (auto& stage : pipeline_.stages) {
      for (auto& node_body : stage->nodes) {
        for (auto& id : node_body->node->tensor_ids) {
          if (id == tensor_id) {
            tensor_id_to_node_[tensor_id] = node_body->node;
            return node_body->node;
          }
        }
      }
    }
  }
  DLOG_ERROR("Tensor {} not found in tensor id to node map", tensor_id);
  return nullptr;
}

NodeBodyPtr ArcherTopologyHandle::GetNodeBodyFromCorrID(
    const std::uint64_t& correlation_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::uint64_t high_corr_id =
      correlation_id >> 32;  // For children in the same level
  std::uint64_t low_corr_id =
      correlation_id & 0xFFFFFFFF;  // For model inference pipeline

  bool is_last_node = (0xFFFFFFFF == high_corr_id);
  if (is_last_node) {
    high_corr_id = 0;  // reset to 0 avoid miss use
  }

  auto stage = pipeline_.stages[low_corr_id];
  auto node_body = stage->nodes[high_corr_id];

  return node_body;
}

// Sparse cache ratio: fraction of *total* device HBM reserved for the LRU
// expert-weight cache. Decoupled from device_memory_ratio, which is the
// overall archer pool budget. Default 0.4 matches the plan
// (cache_capacity_per_rank = floor(0.4 × HBM / expert_bytes)).
// Override with env MOE_INFINITY_SPARSE_RATIO in [0,1].
static double _sparse_cache_ratio() {
  const char* env = std::getenv("MOE_INFINITY_SPARSE_RATIO");
  if (env && *env) {
    char* endp = nullptr;
    double v = std::strtod(env, &endp);
    if (endp != env && v > 0.0 && v <= 1.0) return v;
  }
  return 0.4;
}

std::int64_t ArcherTopologyHandle::GetSparseCacheLimit(
    const torch::Device& device) {
  // Explicit byte override always wins.
  const char* env = std::getenv("MOE_INFINITY_SPARSE_BYTES");
  if (env && *env && device.is_cuda()) {
    char* endp = nullptr;
    long long v = std::strtoll(env, &endp, 10);
    if (endp != env && v > 0) {
      return static_cast<std::int64_t>(v);
    }
  }

  // CUDA path: decoupled ratio. Returns HBM × sparse_cache_ratio regardless
  // of dense_cache_size or device_memory_ratio. This is what the plan
  // intends — device_memory_ratio bounds the overall archer pool while
  // sparse_cache_ratio caps the expert LRU specifically, leaving the
  // remaining (device_memory_ratio - sparse_cache_ratio) × HBM for
  // dense weights, KV cache, activations, and MoELayer pre-alloc buffers.
  if (device.is_cuda()) {
    std::int64_t total_hbm =
        static_cast<std::int64_t>(GetTotalDeviceMemory(device.index()));
    return static_cast<std::int64_t>(total_hbm * _sparse_cache_ratio());
  }

  // Host path: unchanged — full pool minus dense.
  std::int64_t dense_cache_size = 0;
  for (auto& stage : pipeline_.stages) {
    for (auto& node_body : stage->nodes) {
      if (stage->is_sparse) continue;
      if (node_body->node->device == device) {
        dense_cache_size += node_body->node->byte_size;
      }
    }
  }
  std::int64_t device_size_limit = kHostMemoryPool->GetMemoryCapacity();
  assert(device_size_limit > dense_cache_size);
  return device_size_limit - dense_cache_size;
}

std::tuple<std::size_t, std::size_t>
ArcherTopologyHandle::GetNumLayersAndExperts() {
  std::lock_guard<std::mutex> lock(mutex_);
  int num_layers = 0;
  int num_experts = 0;
  for (auto& stage : pipeline_.stages) {
    if (stage->is_sparse) {
      num_layers += 1;
      num_experts = stage->nodes.size();
    }
  }
  return std::make_tuple(num_layers, num_experts);
}

// CPU, GPU -> DISK
// Moves tensors from CPU/GPU to disk.
void SetModuleDisk(std::vector<TensorID>& tensor_ids) {
  // DLOG_TRACE("SetModuleDisk {} tensors", tensor_ids.size());
  for (const auto& tensor_id : tensor_ids) {
    // void* old_ptr = kTensorIndex->find(tensor_id)->second.tensor.data_ptr();
    auto it = kTensorIndex->find(tensor_id);

    at::TensorOptions options;
    options = options.device(torch::kCPU);
    options = options.dtype(it->second.tensor.dtype());
    auto tensor = torch::zeros({1}, options);
    it->second.tensor.set_data(tensor);
  }
}

std::mutex kReadMutex;

// Forward decl from memory_pool.cpp
extern bool _numa_shm_is_follower();
extern bool _numa_shm_enabled();

// DISK -> CPU
void SetModuleMemoryFromDisk(std::vector<TensorID>& tensor_ids, void* host_ptr,
                             bool on_demand) {
  // On NUMA-shared follower processes, data was populated by the leader
  // (same offsets via bump allocator). Skip the disk read.
  const bool skip_read = _numa_shm_enabled() && _numa_shm_is_follower();
  std::int64_t param_size = 0;
  for (const auto& tensor_id : tensor_ids) {
    if (!skip_read) {
      kArcherTensorHandle->ReadTensor(
          tensor_id, (void*)((char*)host_ptr + param_size), on_demand);
    }
    auto it = kTensorIndex->find(tensor_id);
    auto options = torch::TensorOptions()
                       .dtype(it->second.options.dtype())
                       .layout(it->second.options.layout())
                       .device(torch::kCPU)
                       .requires_grad(it->second.options.requires_grad())
                       .pinned_memory(it->second.options.pinned_memory());

    DLOG_TRACE("SetModuleMemoryFromDisk tensor {}", it->second.DebugString());
    auto tensor_tmp =
        torch::from_blob((void*)((char*)host_ptr + param_size),
                         it->second.shape, DoNothingDeleter<void>{}, options);
    if (!it->second.tensor.defined()) {
      it->second.tensor = torch::zeros({1}, options);
    }
    it->second.tensor.set_data(tensor_tmp);
    std::int64_t size_aligned =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    param_size += size_aligned;
  }
}

// DISK (already read) -> CPU views only (no disk read; buffer pre-filled)
void SetModuleMemoryFromDisk_Views(std::vector<TensorID>& tensor_ids,
                                   void* host_ptr) {
  std::int64_t param_size = 0;
  for (const auto& tensor_id : tensor_ids) {
    auto it = kTensorIndex->find(tensor_id);
    auto options = torch::TensorOptions()
                       .dtype(it->second.options.dtype())
                       .layout(it->second.options.layout())
                       .device(torch::kCPU)
                       .requires_grad(it->second.options.requires_grad())
                       .pinned_memory(it->second.options.pinned_memory());

    DLOG_TRACE("SetModuleMemoryFromDisk_Views tensor {}",
               it->second.DebugString());
    auto tensor_tmp =
        torch::from_blob((void*)((char*)host_ptr + param_size),
                         it->second.shape, DoNothingDeleter<void>{}, options);
    if (!it->second.tensor.defined()) {
      it->second.tensor = torch::zeros({1}, options);
    }
    it->second.tensor.set_data(tensor_tmp);
    std::int64_t size_aligned =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    param_size += size_aligned;
  }
}

// CPU -> GPU
void SetModuleCudaMemoryFromCPU(std::vector<TensorID>& tensor_ids,
                                void* device_ptr, const torch::Device& device) {
  // DLOG_TRACE("SetModuleCudaMemoryFromCPU {} tensors", tensor_ids.size());
  std::int64_t param_size = 0;
  for (const auto& tensor_id : tensor_ids) {
    auto it = kTensorIndex->find(tensor_id);
    DLOG_TRACE("SetModuleCudaMemoryFromCPU tensor {} -> {}",
               it->second.DebugString(), device.str());
    auto tensor_options = torch::TensorOptions()
                              .dtype(it->second.options.dtype())
                              .layout(it->second.options.layout())
                              .device(device)
                              .requires_grad(it->second.options.requires_grad())
                              .pinned_memory(false);
    it->second.tensor.set_data(
        torch::from_blob((char*)device_ptr + param_size, it->second.shape,
                         DoNothingDeleter<void>{}, tensor_options));
    std::int64_t size_aligned =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    param_size += size_aligned;
  }
  // DLOG_TRACE("SetModuleCudaMemoryFromCPU {} tensors done",
  // tensor_ids.size());
}

// GPU -> CPU
void SetModuleMemoryFromCuda(std::vector<TensorID>& tensor_ids,
                             void* host_ptr) {
  std::int64_t param_size = 0;
  for (const auto& tensor_id : tensor_ids) {
    // void* old_ptr = kTensorIndex->find(tensor_id)->second.tensor.data_ptr();

    auto it = kTensorIndex->find(tensor_id);
    DLOG_TRACE("SetModuleMemoryFromCuda tensor {}", it->second.DebugString());
    it->second.tensor.set_data(
        torch::from_blob((char*)host_ptr + param_size, it->second.shape,
                         DoNothingDeleter<void>{}, it->second.options));
    // kArcherTensorHandle->UpdateTensorMap(old_ptr,
    // it->second.tensor.data_ptr());
    std::int64_t size_aligned =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    param_size += size_aligned;
  }
  // DLOG_TRACE("SetModuleMemoryFromCuda {} tensors done", tensor_ids.size());
}
