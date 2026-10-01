// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "memory/pinned_memory_pool.h"

#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

#include "utils/logger.h"

// mlock gate (2026-05-28) — mirrors host_caching_allocator.cpp. The AIO
// chunks are cudaHostRegister'd (already pinned for DMA); the extra mlock
// only adds cgroup-unevictable accounting + RLIMIT_MEMLOCK pressure.
// MOE_INFINITY_DISABLE_MLOCK=1 skips the mlock and its paired munlock.
static bool _mlock_disabled() {
  static const bool disabled = []() {
    const char* e = std::getenv("MOE_INFINITY_DISABLE_MLOCK");
    return e && *e && e[0] != '0';
  }();
  return disabled;
}

PinnedMemoryPool::PinnedMemoryPool(std::size_t chunk_size, int num_chunks)
    : chunk_size_(chunk_size) {
  DLOG_INFO("Creating pinned memory pool: ", num_chunks, " chunks of ",
            chunk_size / 1024, " KB");
  all_chunks_.reserve(num_chunks);
  pinned_registered_.reserve(num_chunks);
  for (int i = 0; i < num_chunks; ++i) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, chunk_size) != 0) {
      DLOG_FATAL("Failed to allocate aligned memory for pinned pool chunk ", i);
    }
    cudaError_t err =
        cudaHostRegister(ptr, chunk_size, cudaHostRegisterDefault);
    bool register_success = (err == cudaSuccess);
    if (!register_success) {
      DLOG_WARN("cudaHostRegister failed for chunk ", i, ": ",
                cudaGetErrorString(err), "; falling back to unpinned");
    } else if (!_mlock_disabled() && mlock(ptr, chunk_size) != 0) {
      DLOG_WARN("mlock failed for pinned pool chunk ", i, ": ",
                std::strerror(errno),
                " (set MOE_INFINITY_DISABLE_MLOCK=1 to skip mlock)");
      cudaHostUnregister(ptr);
      register_success = false;
    }
    all_chunks_.push_back(ptr);
    pinned_registered_.push_back(register_success);
    free_list_.push(ptr);
  }
}

PinnedMemoryPool::~PinnedMemoryPool() {
  for (std::size_t i = 0; i < all_chunks_.size(); ++i) {
    void* ptr = all_chunks_[i];
    if (i < pinned_registered_.size() && pinned_registered_[i]) {
      if (!_mlock_disabled()) munlock(ptr, chunk_size_);
      cudaHostUnregister(ptr);
    }
    free(ptr);
  }
}

void* PinnedMemoryPool::Acquire() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !free_list_.empty(); });
  void* ptr = free_list_.front();
  free_list_.pop();
  return ptr;
}

void PinnedMemoryPool::Release(void* ptr) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    free_list_.push(ptr);
  }
  cv_.notify_one();
}
