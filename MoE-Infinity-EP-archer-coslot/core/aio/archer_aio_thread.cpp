// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
//
// See archer_aio_thread.h for the root-cause analysis of the from_pretrained
// load deadlock this file fixes.

#include "archer_aio_thread.h"

#include <chrono>

#include "utils/logger.h"

// Process-wide AIO stats singleton.
AioGlobalStats& ArcherAioStats() {
  static AioGlobalStats s;
  return s;
}

ArcherAioThread::ArcherAioThread(int thread_id)
    : thread_id_(thread_id), is_running_{false} {
  DLOG_INFO("Create ArcherAioThread for thread: ", thread_id_);
}

ArcherAioThread::~ArcherAioThread() { Stop(); }

void ArcherAioThread::Start() {
  if (is_running_.load()) {
    return;
  }

  is_running_.store(true);
  pending_callbacks_ = 0;
  ArcherAioStats().num_workers.fetch_add(1);
  thread_ = std::thread(&ArcherAioThread::Run, this);
}

void ArcherAioThread::Stop() {
  if (!is_running_.load()) {
    return;
  }

  is_running_.store(false);
  cv_.notify_one();
  thread_.join();
  ArcherAioStats().num_workers.fetch_sub(1);
}

void ArcherAioThread::Enqueue(AioCallback& callback) {
  auto& g = ArcherAioStats();
  const std::uint64_t id = g.enqueued.fetch_add(1) + 1;
  g.last_enqueued_id.store(id);
  g.queue_size.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.push_back(Task{id, std::move(callback)});
    pending_callbacks_.fetch_add(1);
  }
  cv_.notify_one();
}

// Wait until this worker's queued callbacks all finished.  Uses a bounded
// wait_for so a lost completion surfaces as a diagnosable FATAL instead of an
// unkillable hang (the historical from_pretrained symptom).
void ArcherAioThread::Wait() {
  auto& g = ArcherAioStats();
  std::unique_lock<std::mutex> lock(mutex_);
  while (pending_callbacks_.load() != 0) {
    if (done_cv_.wait_for(lock, std::chrono::seconds(30), [this] {
          return pending_callbacks_.load() == 0;
        })) {
      break;  // predicate satisfied
    }
    // ---- timed out: dump progress (item 2) ----
    const int qsize = g.queue_size.load();
    const int idle = g.idle_workers.load();
    const int nw = g.num_workers.load();
    const std::uint64_t enq = g.enqueued.load();
    const std::uint64_t comp = g.completed.load();
    const std::uint64_t fail = g.failed.load();
    DLOG_WARN("[AIO_WAIT] thread=", thread_id_,
              " pending=", pending_callbacks_.load(), " queue_size=", qsize,
              " idle_workers=", idle, "/", nw, " enqueued=", enq,
              " started=", g.started.load(), " completed=", comp,
              " failed=", fail, " last_enqueued_id=", g.last_enqueued_id.load(),
              " last_started_id=", g.last_started_id.load(),
              " last_completed_id=", g.last_completed_id.load());
    // ---- lost-completion FATAL (item 3 + 7): every worker idle, nothing
    // queued, yet the accounting is short → a task vanished without running its
    // completion path.  Better to die loudly with the task ids than hang. ----
    if (qsize == 0 && nw > 0 && idle >= nw && (comp + fail) < enq) {
      DLOG_FATAL(
          "[AIO_WAIT] DEADLOCK detected: queue empty + all ", nw,
          " workers idle, but completed+failed (", comp + fail,
          ") < enqueued (", enq, ").  A callback was lost without completing.  "
          "thread=", thread_id_, " pending=", pending_callbacks_.load(),
          " last_enqueued_id=", g.last_enqueued_id.load(),
          " last_started_id=", g.last_started_id.load(),
          " last_completed_id=", g.last_completed_id.load());
    }
  }
  callbacks_.clear();
}

void ArcherAioThread::Run() {
  auto& g = ArcherAioStats();
  while (is_running_.load()) {
    Task task;
    bool have_task = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      g.idle_workers.fetch_add(1);
      cv_.wait(lock,
               [this] { return !callbacks_.empty() || !is_running_.load(); });
      g.idle_workers.fetch_sub(1);
      if (!is_running_.load() && callbacks_.empty()) {
        break;
      }
      if (callbacks_.empty()) {
        continue;
      }
      task = std::move(callbacks_.front());
      callbacks_.pop_front();
      have_task = true;
    }
    if (!have_task) continue;

    g.queue_size.fetch_sub(1);
    g.started.fetch_add(1);
    g.last_started_id.store(task.id);

    // Scope guard (item 4): GUARANTEES the completion accounting + pending--
    // (UNDER mutex_, so Wait()'s predicate check can't miss it → no lost
    // wakeup, item b) + done_cv_ notify on EVERY exit path — normal return,
    // non-zero error code, or thrown exception.
    bool ok = false;
    struct CompletionGuard {
      ArcherAioThread* self;
      AioGlobalStats* g;
      std::uint64_t id;
      bool* ok;
      ~CompletionGuard() {
        {
          std::lock_guard<std::mutex> lock(self->mutex_);
          if (*ok) {
            g->completed.fetch_add(1);
          } else {
            g->failed.fetch_add(1);
          }
          g->last_completed_id.store(id);
          self->pending_callbacks_.fetch_sub(1);
        }
        self->done_cv_.notify_all();
      }
    } guard{this, &g, task.id, &ok};

    try {
      int ret = task.cb ? task.cb() : 0;
      ok = (ret == 0);
      if (!ok) {
        DLOG_WARN("[AIO] task_id=", task.id, " thread=", thread_id_,
                  " callback returned non-zero ret=", ret);
      }
    } catch (const std::exception& e) {
      ok = false;
      DLOG_WARN("[AIO] task_id=", task.id, " thread=", thread_id_,
                " callback threw: ", e.what());
    } catch (...) {
      ok = false;
      DLOG_WARN("[AIO] task_id=", task.id, " thread=", thread_id_,
                " callback threw unknown exception");
    }
    // guard fires here → accounting + pending-- + notify, regardless of path.
  }
}
