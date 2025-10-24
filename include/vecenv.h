#ifndef VECENV_H
#define VECENV_H

#include "rlenv.h"
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <optional>

// Thread-safe channel for inter-thread communication
template<typename T>
class Channel {
public:
    Channel(size_t capacity = 1) : capacity_(capacity), closed_(false) {}
    
    // Non-blocking send - returns false if channel is full
    bool try_send(T&& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() >= capacity_) {
            return false;
        }
        queue_.push(std::move(value));
        cv_.notify_one();
        return true;
    }
    
    // Blocking send - waits until space is available
    void send(T&& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return closed_ || queue_.size() < capacity_; });
        if (closed_) return;
        queue_.push(std::move(value));
        cv_.notify_one();
    }
    
    // Non-blocking receive - returns empty optional if no data
    std::optional<T> try_recv() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop();
        cv_.notify_one();
        return value;
    }
    
    // Blocking receive - waits until data is available
    std::optional<T> recv() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop();
        cv_.notify_one();
        return value;
    }
    
    void close() {
        std::unique_lock<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }
    
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t capacity_;
    bool closed_;
};

// Commands that can be sent to environment threads
enum class EnvCommand {
    RESET,
    STEP,
    SHUTDOWN
};

// Request structure for environment commands
struct EnvRequest {
    EnvCommand command;
    std::array<int, 4> actions; // Only used for STEP command
};

// Response structure from environment threads
struct EnvResponse {
    std::array<std::array<float, 138>, 4> observations;
    float reward;
    bool terminated;
    bool success; // Whether the operation completed successfully
};

// Worker thread that manages a single RLEnv instance
class EnvWorker {
public:
    EnvWorker(int worker_id);
    ~EnvWorker();
    
    // Delete copy/move constructors
    EnvWorker(const EnvWorker&) = delete;
    EnvWorker& operator=(const EnvWorker&) = delete;
    EnvWorker(EnvWorker&&) = delete;
    EnvWorker& operator=(EnvWorker&&) = delete;
    
    // Send a request to this worker (non-blocking)
    bool send_request(EnvRequest&& request);
    
    // Try to receive a response from this worker (non-blocking)
    std::optional<EnvResponse> try_recv_response();
    
    // Blocking receive response
    std::optional<EnvResponse> recv_response();
    
    // Check if worker is ready for new request
    bool is_ready() const { return ready_.load(); }
    
    void shutdown();
    
private:
    void worker_loop();
    
    int worker_id_;
    std::unique_ptr<RLEnv> env_;
    std::thread thread_;
    
    // Channels for bidirectional communication
    Channel<EnvRequest> request_channel_;
    Channel<EnvResponse> response_channel_;
    
    std::atomic<bool> running_;
    std::atomic<bool> ready_;
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
    
    // Reset all environments and return observations (optional - envs auto-reset on termination)
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
    // Environments automatically reset when terminated
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
    
    // Track which workers have pending async operations
    std::vector<bool> async_pending_;
};

#endif // VECENV_H