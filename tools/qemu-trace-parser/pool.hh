#pragma once

#include <concepts>
#include <functional>
#include <future>
#include <stop_token>
#include <thread>
#include <vector>

namespace cheri {

/**
 * Worker descriptor in the pool
 */
struct WorkerState {
  std::thread thr;
};

class WorkerPool {
public:
  WorkerPool() = default;
  WorkerPool(const WorkerPool &other) = delete;
  ~WorkerPool() = default;

  // Run a function in a worker thread
  template <std::invocable<std::stop_token> F>
  std::future<std::invoke_result_t<F, std::stop_token>> submit(F fn) {
    using Result = std::invoke_result_t<F, std::stop_token>;
    std::packaged_task<Result(std::stop_token)> task(fn);
    auto result = task.get_future();

    WorkerState ws({std::thread(std::move(task), stopctrl_.get_token())});
    workers_.emplace_back(std::move(ws));
    return result;
  }

  // Number of pending tasks
  std::size_t size() const { return workers_.size(); }

  // Cancel all pending tasks
  void cancel() { stopctrl_.request_stop(); }

  // Wait for all tasks to complete and drain the workers queue
  void drain() {
    while (!workers_.empty()) {
      WorkerState ws = std::move(workers_.back());
      ws.thr.join();
      workers_.pop_back();
    }
    // XXX replace the stop source?
  }

private:
  std::vector<WorkerState> workers_;
  std::stop_source stopctrl_;
};

} /* namespace cheri */
