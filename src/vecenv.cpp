#include "vecenv.h"
#include <iostream>
#include <algorithm>
#include <chrono>

EnvWorker::EnvWorker(int worker_id, WorkerState* state, ObservationBuffer* obs_buffer)
    : worker_id_(worker_id),
      state_(state),
      obs_buffer_(obs_buffer),
      env_(nullptr),
      running_(true)
{
    // Start the worker thread
    thread_ = std::thread(&EnvWorker::worker_loop, this);
}

EnvWorker::~EnvWorker() {
    shutdown();
}

void EnvWorker::shutdown() {
    if (running_.load()) {
        running_.store(false);
        
        // Send shutdown command
        state_->command.store(EnvCommand::SHUTDOWN, std::memory_order_release);
        
        // Wait for thread to finish
        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

void EnvWorker::worker_loop() {
    try {
        // Initialize environment in this thread for thread safety
        env_ = std::make_unique<RLEnv>();
        
        // Auto-reset on initialization
        env_->reset(obs_buffer_->observations);
        obs_buffer_->reward = 0.0f;
        obs_buffer_->terminated = false;
        
        // Mark as ready
        state_->ready.store(true, std::memory_order_release);
        
        while (running_.load(std::memory_order_relaxed)) {
            // Wait for a command using atomic operations
            EnvCommand cmd = state_->command.load(std::memory_order_acquire);
            
            if (cmd == EnvCommand::IDLE) {
                // No work to do, yield CPU
                std::this_thread::yield();
                continue;
            }
            
            // Mark as busy
            state_->ready.store(false, std::memory_order_relaxed);
            
            switch (cmd) {
                case EnvCommand::RESET: {
                    env_->reset(obs_buffer_->observations);
                    obs_buffer_->reward = 0.0f;
                    obs_buffer_->terminated = false;
                    break;
                }
                
                case EnvCommand::STEP: {
                    env_->step(
                        state_->actions,
                        obs_buffer_->observations,
                        obs_buffer_->reward,
                        obs_buffer_->terminated
                    );
                    
                    // Auto-reset if environment terminated
                    if (obs_buffer_->terminated) {
                        env_->reset(obs_buffer_->observations);
                        // Keep terminated flag true so caller knows episode ended
                    }
                    break;
                }
                
                case EnvCommand::SHUTDOWN: {
                    running_.store(false);
                    state_->ready.store(true, std::memory_order_release);
                    return;
                }
                
                case EnvCommand::IDLE:
                    break;
            }
            
            // Mark as ready and clear command
            state_->command.store(EnvCommand::IDLE, std::memory_order_release);
            state_->ready.store(true, std::memory_order_release);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Worker " << worker_id_ << " error: " << e.what() << std::endl;
        running_.store(false);
    }
}

// ============================================================================
// VecEnv Implementation
// ============================================================================

VecEnv::VecEnv(int num_envs)
    : num_envs_(num_envs),
      async_pending_(num_envs, false)
{
    if (num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }
    
    // Allocate cache-aligned buffers
    obs_buffers_ = std::unique_ptr<ObservationBuffer[]>(new ObservationBuffer[num_envs]);
    worker_states_ = std::unique_ptr<WorkerState[]>(new WorkerState[num_envs]);
    
    // Create all worker threads
    workers_.reserve(num_envs);
    for (int i = 0; i < num_envs; ++i) {
        workers_.push_back(std::make_unique<EnvWorker>(i, &worker_states_[i], &obs_buffers_[i]));
    }
    
    // Wait for all workers to initialize
    bool all_ready = false;
    while (!all_ready) {
        all_ready = true;
        for (int i = 0; i < num_envs; ++i) {
            if (!worker_states_[i].ready.load(std::memory_order_acquire)) {
                all_ready = false;
                break;
            }
        }
        if (!all_ready) {
            std::this_thread::yield();
        }
    }
}

VecEnv::~VecEnv() {
    // Workers will be shut down by their destructors
}

std::vector<std::array<std::array<float, 138>, 4>> VecEnv::reset() {
    std::vector<std::array<std::array<float, 138>, 4>> all_observations;
    all_observations.reserve(num_envs_);
    
    // Send reset commands to all workers simultaneously
    for (int i = 0; i < num_envs_; ++i) {
        // Wait for worker to be ready
        while (!worker_states_[i].ready.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        // Issue reset command
        worker_states_[i].command.store(EnvCommand::RESET, std::memory_order_release);
    }
    
    // Wait for all workers to complete
    for (int i = 0; i < num_envs_; ++i) {
        // Wait for command to be cleared (indicating completion)
        while (worker_states_[i].command.load(std::memory_order_acquire) != EnvCommand::IDLE) {
            std::this_thread::yield();
        }
        
        // Copy observation from shared buffer
        all_observations.push_back(obs_buffers_[i].observations);
    }
    
    return all_observations;
}

std::tuple<
    std::vector<std::array<std::array<float, 138>, 4>>,
    std::vector<float>,
    std::vector<bool>
> VecEnv::step(const std::vector<std::array<int, 4>>& actions) {
    if (actions.size() != static_cast<size_t>(num_envs_)) {
        throw std::invalid_argument(
            "Number of action arrays (" + std::to_string(actions.size()) + 
            ") must match num_envs (" + std::to_string(num_envs_) + ")"
        );
    }
    
    step_async(actions);
    return step_wait();
}

void VecEnv::step_async(const std::vector<std::array<int, 4>>& actions) {
    if (actions.size() != static_cast<size_t>(num_envs_)) {
        throw std::invalid_argument(
            "Number of action arrays (" + std::to_string(actions.size()) + 
            ") must match num_envs (" + std::to_string(num_envs_) + ")"
        );
    }
    
    // Send step commands to all workers simultaneously
    for (int i = 0; i < num_envs_; ++i) {
        // Wait for worker to be ready (should already be ready in normal operation)
        while (!worker_states_[i].ready.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        // Copy actions to worker's buffer
        worker_states_[i].actions = actions[i];
        
        // Issue step command
        worker_states_[i].command.store(EnvCommand::STEP, std::memory_order_release);
        
        async_pending_[i] = true;
    }
}

std::tuple<
    std::vector<std::array<std::array<float, 138>, 4>>,
    std::vector<float>,
    std::vector<bool>
> VecEnv::step_wait() {
    std::vector<std::array<std::array<float, 138>, 4>> all_observations;
    std::vector<float> all_rewards;
    std::vector<bool> all_dones;
    
    all_observations.reserve(num_envs_);
    all_rewards.reserve(num_envs_);
    all_dones.reserve(num_envs_);
    
    // Wait for all workers to complete and collect results
    for (int i = 0; i < num_envs_; ++i) {
        if (!async_pending_[i]) {
            throw std::runtime_error(
                "step_wait called without corresponding step_async for worker " + 
                std::to_string(i)
            );
        }
        
        // Wait for command to be cleared (indicating completion)
        while (worker_states_[i].command.load(std::memory_order_acquire) != EnvCommand::IDLE) {
            std::this_thread::yield();
        }
        
        // Copy results from shared buffers
        all_observations.push_back(obs_buffers_[i].observations);
        all_rewards.push_back(obs_buffers_[i].reward);
        all_dones.push_back(obs_buffers_[i].terminated);
        
        async_pending_[i] = false;
    }
    
    return std::make_tuple(
        std::move(all_observations),
        std::move(all_rewards),
        std::move(all_dones)
    );
}