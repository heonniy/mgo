// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// 2026-05-29 root-cause fix for the from_pretrained load deadlock.
// =================================================================
// Symptom (235B EP=4, intermittent): a NUMA-leader rank hangs in
// from_pretrained — main thread futex_wait forever, ALL AIO worker threads
// idle, zero io/lustre-wait threads → all reads finished but the completion
// accounting was lost.
//
// Two defects in the old ArcherAioThread::Run / Wait:
//   (a) callback() ran WITHOUT try/catch and its int return (error code) was
//       discarded; an exception skipped the pending-- and notify → counter
//       stuck > 0 forever.
//   (b) pending_callbacks_ was decremented and done_cv_ notified OUTSIDE the
//       mutex_ that Wait() holds for its predicate check → classic lost
//       wakeup (notify fires in the window after Wait's pred-check-false but
//       before it registers as a waiter).
//
// Fix: a scope guard performs the accounting (completed/failed) + pending--
// UNDER mutex_ + notify on EVERY path (success / non-zero return / exception),
// global task counters expose progress, and Wait() uses a 30s wait_for + an
// [AIO_WAIT] dump and FATALs on the lost-completion signature instead of
// hanging forever.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <thread>

#include "archer_aio_utils.h"

// Process-wide AIO task accounting (shared across all worker threads), used for
// progress diagnostics and the lost-completion FATAL.
struct AioGlobalStats {
  std::atomic<std::uint64_t> enqueued{0};
  std::atomic<std::uint64_t> started{0};
  std::atomic<std::uint64_t> completed{0};
  std::atomic<std::uint64_t> failed{0};
  std::atomic<std::uint64_t> last_enqueued_id{0};
  std::atomic<std::uint64_t> last_started_id{0};
  std::atomic<std::uint64_t> last_completed_id{0};
  std::atomic<int> queue_size{0};    // enqueued and not yet popped by a worker
  std::atomic<int> idle_workers{0};  // workers blocked waiting for work
  std::atomic<int> num_workers{0};   // total live worker threads
};
AioGlobalStats& ArcherAioStats();

class ArcherAioThread {
 public:
  explicit ArcherAioThread(int thread_id);
  ~ArcherAioThread();

  void Start();
  void Stop();

  void Enqueue(AioCallback& callback);
  void Wait();

 private:
  void Run();

  struct Task {
    std::uint64_t id;
    AioCallback cb;
  };

 private:
  int thread_id_;
  std::thread thread_;
  std::atomic<bool> is_running_;

  std::list<Task> callbacks_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::atomic<int> pending_callbacks_;
};
