#ifndef VECENV_H
#define VECENV_H

#include "rlenv.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <array>

// Cache line size for alignment (avoid false sharing)
constexpr size_t CACHE_LINE_SIZE = 64;

// Commands that can be sent to environment threads
enum class EnvCommand : int {
    IDLE = 0,
    RESET = 1,
    STEP = 2,
    SHUTDOWN = 3
};

// Aligned observation buffer to avoid false sharing between threads
struct alignas(CACHE_LINE_SIZE) ObservationBuffer {
    std::array<std::array<float, 138>, 4> observations;
    float reward;
    bool terminated;
};

// Worker thread state - each worker has its own cache line
struct alignas(CACHE_LINE_SIZE) WorkerState {
    std::atomic<EnvCommand> command{EnvCommand::IDLE};
    std::atomic<bool> ready{false};
    std::array<int, 4> actions;
};

// Worker thread that manages a single RLEnv instance
class EnvWorker {
public:
    EnvWorker(int worker_id, WorkerState* state, ObservationBuffer* obs_buffer);
    ~EnvWorker();
    
    // Delete copy/move constructors
    EnvWorker(const EnvWorker&) = delete;
    EnvWorker& operator=(const EnvWorker&) = delete;
    EnvWorker(EnvWorker&&) = delete;
    EnvWorker& operator=(EnvWorker&&) = delete;
    
    void shutdown();
    
private:
    void worker_loop();
    
    int worker_id_;
    WorkerState* state_;
    ObservationBuffer* obs_buffer_;
    std::unique_ptr<RLEnv> env_;
    std::thread thread_;
    std::atomic<bool> running_;
};

// Vectorized environment that manages multiple RLEnv instances in parallel
class VecEnv {
public:
    explicit VecEnv(int num_envs);
    ~VecEnv();
    
    // Delete copy/move constructors
    VecEnv(const VecEnv&) = delete;
    VecEnv& operator=(const VecEnv&) = delete;
    VecEnv(VecEnv&&) = delete;
    VecEnv& operator=(VecEnv&&) = delete;
    
    // Reset all environments and return observations
    // Environments are automatically initialized on construction
    // Shape: [num_envs, 4, 138]
    std::vector<std::array<std::array<float, 138>, 4>> reset();
    
    // Step all environments with given actions
    // Environments automatically reset when terminated (observations will be from reset)
    // actions shape: [num_envs, 4]
    // Returns: (observations, rewards, dones)
    //   observations shape: [num_envs, 4, 138] (reset obs if done[i] == true)
    //   rewards shape: [num_envs]
    //   dones shape: [num_envs] (true indicates episode just ended and env was reset)
    std::tuple<
        std::vector<std::array<std::array<float, 138>, 4>>,
        std::vector<float>,
        std::vector<bool>
    > step(const std::vector<std::array<int, 4>>& actions);
    
    // Async step - send all step commands without waiting for results
    void step_async(const std::vector<std::array<int, 4>>& actions);
    
    // Wait for all async step results
    std::tuple<
        std::vector<std::array<std::array<float, 138>, 4>>,
        std::vector<float>,
        std::vector<bool>
    > step_wait();
    
    // Get number of environments
    int num_envs() const { return num_envs_; }
    
    // Get number of agents per environment
    static constexpr int num_agents() { return 4; }
    
    // Get observation size per agent
    static constexpr int obs_size() { return 138; }
    
private:
    int num_envs_;
    std::vector<std::unique_ptr<EnvWorker>> workers_;
    
    // Pre-allocated, cache-aligned buffers for observations
    std::unique_ptr<ObservationBuffer[]> obs_buffers_;
    
    // Pre-allocated, cache-aligned worker states
    std::unique_ptr<WorkerState[]> worker_states_;
    
    // Track which workers have pending async operations
    std::vector<bool> async_pending_;
};

#endif // VECENV_H