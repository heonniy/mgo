// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "memory_pool.h"

#include "common/types.h"
#include "host_caching_allocator.h"
#include "memory/stream_pool.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"

#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cstdlib>
#include <c10/cuda/CUDAGuard.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <atomic>
#include <fstream>
#include <mutex>
#include <string>

std::unique_ptr<HostMemoryPool> kHostMemoryPool(nullptr);
std::unique_ptr<DeviceMemoryPool> kDeviceMemoryPool(nullptr);

std::size_t GetTotalSystemMemory() {
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

// NUMA-shared pinned host memory: Python sets up a memfd shared across
// processes on the same NUMA, mmaps it, and calls cudaHostRegister. The
// base address is passed via MOE_INFINITY_SHM_BASE_PTR env. archer's
// HostMemoryPool then bump-allocates into that shared region instead of
// per-process pinned memory.
// Both leader and follower processes deterministically iterate sparse_nodes
// in the same order, so bump offsets line up across processes for the same
// tensor. Leader's SetModuleMemoryFromDisk populates the bytes; follower
// (MOE_INFINITY_NUMA_FOLLOWER=1) skips the disk read in model_topology.cpp.
struct NumaSharedShmCfg {
  bool enabled = false;
  char* base = nullptr;
  std::int64_t size = 0;
  std::atomic<std::int64_t> bump_offset{0};
  // non-copyable/non-movable because of std::atomic; populated in-place.
};
static NumaSharedShmCfg& _shm_cfg() {
  static NumaSharedShmCfg cfg;  // default-constructed (disabled, zero bump)
  static std::once_flag once;
  std::call_once(once, []() {
    const char* env_base = std::getenv("MOE_INFINITY_SHM_BASE_PTR");
    const char* env_size = std::getenv("MOE_INFINITY_SHM_SIZE");
    if (env_base && env_size && *env_base && *env_size) {
      char* endp = nullptr;
      unsigned long long base_ull = std::strtoull(env_base, &endp, 0);
      if (endp != env_base && base_ull != 0) {
        long long sz = std::strtoll(env_size, nullptr, 0);
        if (sz > 0) {
          cfg.base = reinterpret_cast<char*>(base_ull);
          cfg.size = sz;
          cfg.enabled = true;
          DLOG_INFO("NUMA-shared shm enabled: base=", env_base,
                    " size=", sz, " bytes");
        }
      }
    }
  });
  return cfg;
}

// Forward-declared so AllocateMemory can reference it; the actual definition
// is below (after AllocateMemory). thread_local: per-process per-thread.
static thread_local bool t_use_shm_alloc;

// Forward declarations — definitions live below AllocateMemory but the
// admission-control branch (added 2026-05-28) references them for diagnostics.
static double _host_memory_ratio();
static std::int64_t _read_cgroup_memory_max();
static int _read_world_size();

void* HostMemoryPool::AllocateMemory(const std::size_t key,
                                     const std::int64_t size,
                                     const torch::Device& device) {
  assert(device.is_cpu());
  std::unique_lock<std::mutex> lock(mutex_);
  if (allocated_id_.find(key) != allocated_id_.end()) {
    DLOG_ERROR("PreAllocateMemory failed, already allocated ", key);
    return allocated_id_[key];
  }
  auto& shm = _shm_cfg();
  void* data_ptr = nullptr;
  if (shm.enabled && t_use_shm_alloc) {
    // Bump-allocate into the NUMA-shared region. Both leader and follower
    // call this in the same order across processes (archer's sparse_nodes
    // iteration is deterministic), so the same `key` lands at the same offset.
    //
    // OVERFLOW SAFETY (2026-05-27): fetch_add returns the *prior* offset and
    // unconditionally commits the increment.  If the result would exceed the
    // shm region we MUST roll the increment back, otherwise a subsequent
    // allocation (or a DLOG_FATAL that doesn't actually abort) would land at
    // an offset outside the mapped region → SIGSEGV on first byte access.
    std::int64_t off = shm.bump_offset.fetch_add(size);
    if (off + size > shm.size) {
      shm.bump_offset.fetch_sub(size);  // rollback
      DLOG_FATAL("NUMA shm bump overflow: off=", off, " size=", size,
                 " cap=", shm.size);
    }
    data_ptr = static_cast<void*>(shm.base + off);
  } else {
    // Admission control (2026-05-28): the host pool capacity is computed
    // cgroup-aware in the ctor (_effective_host_capacity), but allocations
    // previously ignored it and called cudaHostAlloc unconditionally — a
    // runaway (wrong ratio, WORLD_SIZE unset, model larger than the budget)
    // walked straight into the cgroup limit and the kernel OOM-killer sent
    // SIGKILL (exit=137) with NO diagnostic.  Refuse here instead, naming the
    // capacity / current usage / inputs so the operator sees *why* before the
    // process dies.  shm-mode allocations skip this (the memfd is sized and
    // bounds-checked by the bump path above + Python's cgroup sanity check).
    if (free_memory_ - size < 0) {
      DLOG_ERROR("HostMemoryPool admission denied: requested=", size,
                 " free=", free_memory_, " capacity=", memory_capacity_,
                 " key=", key, " ratio=", _host_memory_ratio(),
                 " world_size=", _read_world_size(),
                 " cgroup_max=", _read_cgroup_memory_max(),
                 " — host pool would exceed its cgroup-aware budget. Lower "
                 "MOE_INFINITY_HOST_MEMORY_RATIO, raise the cgroup memory "
                 "limit, or verify WORLD_SIZE propagation to this rank.");
      throw std::runtime_error(
          "HostMemoryPool exhausted (cgroup-aware budget exceeded); see "
          "DLOG_ERROR above for capacity/usage. Aborting deterministically "
          "instead of risking a silent OOM-kill (exit=137).");
    }
    auto allocator = c10::HostCachingAllocator::get();
    data_ptr = allocator->allocate(size);
  }
  allocated_id_.insert(std::make_pair(key, data_ptr));
  // Accounting (2026-05-27 fix): FreeMemory adds size back, but AllocateMemory
  // never subtracted → free_memory_ only ever increased, making GetFreeMemory
  // return a meaningless value.  Decrement here to keep the counter honest.
  // Note: shm-mode allocations are charged separately (the memfd is owned by
  // Python) but we still track them here so a per-process bookkeeping query
  // reflects the bytes this process *requested* — symmetric with the FreeMemory
  // accounting at line ~142 which credits the size back regardless of shm.
  free_memory_ -= size;
  return data_ptr;
}

// Exposed to other TUs (model_topology.cpp) so it can decide whether to
// skip the disk-read step on follower processes.
bool _numa_shm_is_follower() {
  const char* env = std::getenv("MOE_INFINITY_NUMA_FOLLOWER");
  return env && *env && env[0] != '0';
}
bool _numa_shm_enabled() { return _shm_cfg().enabled; }

// Scope flag accessors. t_use_shm_alloc is the forward-declared thread_local
// near the top of this file. Set by model_topology.cpp around sparse-load.
void _set_use_shm_alloc(bool v) { t_use_shm_alloc = v; }
bool _get_use_shm_alloc() { return t_use_shm_alloc; }

int HostMemoryPool::FreeMemory(const std::size_t key, void* data,
                               const std::int64_t size,
                               const torch::Device& device) {
  assert(device.is_cpu());
  std::unique_lock<std::mutex> lock(mutex_);
  if (allocated_id_.find(key) == allocated_id_.end()) {
    DLOG_ERROR("FreeMemory failed, not found ", key);
    return -1;
  }
  allocated_id_.erase(key);
  // shm-mode allocations are owned by Python (memfd-backed); don't free
  // them here. Detect by pointer range: shm allocations have addresses
  // inside [shm.base, shm.base + shm.size).
  auto& shm = _shm_cfg();
  bool is_shm_ptr = shm.enabled && data != nullptr &&
                    static_cast<char*>(data) >= shm.base &&
                    static_cast<char*>(data) < shm.base + shm.size;
  if (data != nullptr && !is_shm_ptr) {
    auto allocator = c10::HostCachingAllocator::get();
    allocator->free(data);
  }
  free_memory_ += size;
  return 0;
}

// Env override: MOE_INFINITY_HOST_MEMORY_RATIO. Each rank accounts for its
// own host-RAM share — with N ranks sharing a cgroup, we need
// (ratio × system_RAM) × N ≤ cgroup_limit. Default 0.8 (compile-time) works
// for single-rank; multi-rank without env override is now also safe because
// HostMemoryPool init clamps the capacity against the per-rank share of the
// detected cgroup limit (see _effective_host_capacity below).
static double _host_memory_ratio() {
  const char* env = std::getenv("MOE_INFINITY_HOST_MEMORY_RATIO");
  if (env && *env) {
    char* endp = nullptr;
    double v = std::strtod(env, &endp);
    if (endp != env && v > 0.0 && v <= 1.0) return v;
  }
  return static_cast<double>(HOST_MEMORY_RATIO);
}

// cgroup-aware host memory capacity.
// Without this clamp, every rank reports (ratio × system_RAM) as its own
// share; with N ranks in one cgroup the *sum* exceeds memory.max long
// before any single rank notices. Reading the cgroup limit and dividing
// by WORLD_SIZE (with 20% headroom for Python / dense weights / HF
// from_pretrained transient) keeps per-rank reservations within the
// shared limit even when MOE_INFINITY_HOST_MEMORY_RATIO is left at the
// default. cgroup v2 path is /sys/fs/cgroup/memory.max; if the file is
// missing or contains "max", we fall through to the ratio-only value.
static std::int64_t _read_cgroup_memory_max() {
  std::ifstream f("/sys/fs/cgroup/memory.max");
  if (!f.is_open()) {
    std::ifstream f1("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    if (!f1.is_open()) return 0;
    std::string s1;
    std::getline(f1, s1);
    try { return std::stoll(s1); } catch (...) { return 0; }
  }
  std::string s;
  std::getline(f, s);
  if (s.empty() || s == "max") return 0;
  try { return std::stoll(s); } catch (...) { return 0; }
}

static int _read_world_size() {
  const char* env = std::getenv("WORLD_SIZE");
  if (env && *env) {
    char* endp = nullptr;
    long v = std::strtol(env, &endp, 10);
    if (endp != env && v > 0 && v < 4096) return static_cast<int>(v);
  }
  // WORLD_SIZE missing/invalid → assume single rank. With N ranks sharing one
  // cgroup this OVER-estimates the per-rank host budget (cgroup_max / 1 instead
  // of / N), so the sum of all ranks' reservations can exceed memory.max before
  // any single rank's admission control trips → OOM-kill (exit=137). Warn once.
  static std::once_flag warned;
  std::call_once(warned, []() {
    DLOG_ERROR("WORLD_SIZE unset or invalid — assuming 1 for host-pool "
               "cgroup accounting. If this is a multi-rank run the per-rank "
               "budget is over-estimated and risks OOM-kill. Ensure the "
               "launcher exports WORLD_SIZE to every rank (e.g. numactl wrap).");
  });
  return 1;
}

static std::int64_t _effective_host_capacity() {
  std::int64_t ratio_share = static_cast<std::int64_t>(
      GetTotalSystemMemory() * _host_memory_ratio());
  std::int64_t cgroup_max = _read_cgroup_memory_max();
  int ws = _read_world_size();
  if (cgroup_max <= 0 || ws <= 0) return ratio_share;
  // 20% headroom for Python interpreter, HF from_pretrained transient
  // peak (model state dict held in RAM until archer takeover), per-rank
  // dense weights pinned by torch's own allocator, and NUMA-shared shm
  // pages charged once but visible from this rank's perspective.
  std::int64_t per_rank_cgroup = (cgroup_max * 4 / 5) / ws;
  return std::min(ratio_share, per_rank_cgroup);
}

HostMemoryPool::HostMemoryPool()
    : free_memory_(
#ifdef TEST_LIMIT_MEMORY
          10LL * 1024 * 1024 * 1024
#else
          _effective_host_capacity()
#endif
      ) {
  auto pinned_mr_ = c10::HostCachingAllocator::get();
  if (pinned_mr_ == nullptr) {
    DLOG_ERROR("GetHostAllocator failed");
    exit(-1);
  }
  memory_capacity_ = free_memory_;
  DLOG_INFO("HostMemoryPool init: ratio=", _host_memory_ratio(),
            " world_size=", _read_world_size(),
            " cgroup_max=", _read_cgroup_memory_max(),
            " sys_ram=", GetTotalSystemMemory(),
            " free_memory=", free_memory_, " bytes");
}

std::int64_t HostMemoryPool::GetFreeMemory() {
  std::lock_guard<std::mutex> lock(mutex_);
  return free_memory_;
}

std::int64_t HostMemoryPool::GetMemoryCapacity() { return memory_capacity_; }

void* DeviceMemoryPool::AllocateMemory(const std::size_t key,
                                       const std::int64_t size,
                                       const torch::Device& device) {
  int device_id = device.index();
  std::unique_lock<std::mutex> lock(mutex_);
  if (allocated_id_[device_id].find(key) != allocated_id_[device_id].end()) {
    DLOG_ERROR("PreAllocateMemory failed, already allocated ", key);
    return allocated_id_[device_id][key];
  }
  cudaSetDevice(device_id);
  // at::Allocator* allocator = c10::cuda::CUDACachingAllocator::get();
  // auto data_ptr = allocator->raw_allocate(size);
  auto allocator = c10::DeviceCachingAllocator::get(device_id);
  auto data_ptr = allocator->allocate(size);
  free_memory_[device_id] -= size;
  allocated_id_[device_id].insert(std::make_pair(key, data_ptr));
  return data_ptr;
}

int DeviceMemoryPool::FreeMemory(const std::size_t key, void* data,
                                 const std::int64_t size,
                                 const torch::Device& device) {
  int device_id = device.index();
  std::unique_lock<std::mutex> lock(mutex_);
  if (allocated_id_[device_id].find(key) == allocated_id_[device_id].end()) {
    DLOG_ERROR("FreeMemory failed, not found ", key);
    return -1;
  }
  allocated_id_[device_id].erase(key);
  cudaSetDevice(device_id);
  if (data != nullptr) {
    // at::Allocator* allocator = c10::cuda::CUDACachingAllocator::get();
    // allocator->raw_deallocate(data);
    auto allocator = c10::DeviceCachingAllocator::get(device_id);
    allocator->free(data);
  }
  free_memory_[device_id] += size;
  return 0;
}

DeviceMemoryPool::DeviceMemoryPool() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  // c10::cuda::CUDACachingAllocator::init(device_count);

  for (int i = 0; i < device_count; ++i) {
    std::unordered_map<std::uint64_t, void*> allocated_id;
    allocated_id_.emplace_back(allocated_id);
    free_memory_.emplace_back(GetTotalDeviceMemory(i));
    memory_capacity_.emplace_back(free_memory_[i]);
  }
}

std::int64_t DeviceMemoryPool::GetFreeMemory(const torch::Device& device) {
  int device_id = device.index();
  std::lock_guard<std::mutex> lock(mutex_);
  return free_memory_[device_id];
}

std::int64_t DeviceMemoryPool::GetMemoryCapacity(const torch::Device& device) {
  int device_id = device.index();
  return memory_capacity_[device_id];
}

void DeviceMemoryPool::SetMemoryRatio(const double ratio) {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  for (int i = 0; i < device_count; ++i) {
    free_memory_[i] = GetTotalDeviceMemory(i) * ratio;
    memory_capacity_[i] = free_memory_[i];
  }
}
