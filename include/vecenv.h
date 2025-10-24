#pragma once

#include "rlenv.h"
#include <vector>
#include <thread>
#include <memory>
#include <array>
#include <tuple>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;

// Commands for worker threads
enum class WorkerCommand : uint32_t {
    IDLE,
    RESET,
    STEP,
    SHUTDOWN
};

// Cache-aligned data structure to prevent false sharing
struct alignas(CACHE_LINE_SIZE) WorkerData {
    WorkerCommand command{WorkerCommand::IDLE};
    bool work_ready{false};
    bool work_done{true};
    char padding[CACHE_LINE_SIZE - sizeof(WorkerCommand) - 2];
};

class BatchWorker {
public:
    BatchWorker(int worker_id, int num_envs_in_batch);
    ~BatchWorker();

    BatchWorker(const BatchWorker&) = delete;
    BatchWorker& operator=(const BatchWorker&) = delete;
    BatchWorker(BatchWorker&&) = delete;
    BatchWorker& operator=(BatchWorker&&) = delete;

    void reset_async(std::vector<std::array<std::array<float, 138>, 4>>* observations,
                     int start_idx);

    void step_async(const std::vector<std::array<int, 4>>* actions,
                    std::vector<std::array<std::array<float, 138>, 4>>* observations,
                    std::vector<float>* rewards,
                    std::vector<uint8_t>* dones,
                    int start_idx);

    void wait();

    void shutdown();

    int num_envs() const { return num_envs_; }

private:
    void worker_loop();

    int worker_id_;
    int num_envs_;
    std::vector<std::unique_ptr<RLEnv>> envs_;

    // Thread management
    std::thread thread_;
    std::atomic<bool> running_{true};

    // Command synchronization with cache line alignment
    alignas(CACHE_LINE_SIZE) std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    alignas(CACHE_LINE_SIZE) WorkerData worker_data_;

    // Pointers to shared data buffers and starting index
    alignas(CACHE_LINE_SIZE) int start_idx_{0};
    const std::vector<std::array<int, 4>>* actions_{nullptr};
    std::vector<std::array<std::array<float, 138>, 4>>* observations_{nullptr};
    std::vector<float>* rewards_{nullptr};
    std::vector<uint8_t>* dones_{nullptr};
};

class VecEnv {
public:
    explicit VecEnv(int num_envs, int num_threads = 0);
    ~VecEnv();

    VecEnv(const VecEnv&) = delete;
    VecEnv& operator=(const VecEnv&) = delete;
    VecEnv(VecEnv&&) = delete;
    VecEnv& operator=(VecEnv&&) = delete;

    std::vector<std::array<std::array<float, 138>, 4>> reset();

    std::tuple<
        std::vector<std::array<std::array<float, 138>, 4>>,
        std::vector<float>,
        std::vector<uint8_t>
    > step(const std::vector<std::array<int, 4>>& actions);

    int num_envs() const { return num_envs_; }

    int num_threads() const { return num_threads_; }

    static constexpr int num_agents() { return 4; }

    static constexpr int obs_size() { return 138; }

private:
    int num_envs_;
    int num_threads_;
    int envs_per_thread_;

    std::vector<std::unique_ptr<BatchWorker>> workers_;

    // Pre-allocated buffers - cache-aligned via environment distribution
    std::vector<std::array<std::array<float, 138>, 4>> all_observations_;
    std::vector<float> all_rewards_;
    std::vector<uint8_t> all_dones_;
};
