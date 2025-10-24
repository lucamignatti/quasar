#include "vecenv.h"
#include <iostream>
#include <algorithm>
#include <chrono>


EnvWorker::EnvWorker(int worker_id)
    : worker_id_(worker_id),
      env_(nullptr),
      request_channel_(1),  // Buffer size of 1 for requests
      response_channel_(1), // Buffer size of 1 for responses
      running_(true),
      ready_(false)
{
    // Start the worker thread
    thread_ = std::thread(&EnvWorker::worker_loop, this);
}

EnvWorker::~EnvWorker() {
    shutdown();
}

bool EnvWorker::send_request(EnvRequest&& request) {
    return request_channel_.try_send(std::move(request));
}

std::optional<EnvResponse> EnvWorker::try_recv_response() {
    return response_channel_.try_recv();
}

std::optional<EnvResponse> EnvWorker::recv_response() {
    return response_channel_.recv();
}

void EnvWorker::shutdown() {
    if (running_.load()) {
        running_.store(false);
        
        // Send shutdown command
        EnvRequest shutdown_req;
        shutdown_req.command = EnvCommand::SHUTDOWN;
        request_channel_.send(std::move(shutdown_req));
        
        // Close channels
        request_channel_.close();
        response_channel_.close();
        
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
        EnvResponse init_response;
        env_->reset(init_response.observations);
        
        ready_.store(true);
        
        while (running_.load()) {
            // Wait for a request
            auto request_opt = request_channel_.recv();
            if (!request_opt.has_value()) {
                break; // Channel closed
            }
            
            ready_.store(false);
            EnvRequest& request = request_opt.value();
            EnvResponse response;
            response.success = true;
            
            switch (request.command) {
                case EnvCommand::RESET: {
                    env_->reset(response.observations);
                    response.reward = 0.0f;
                    response.terminated = false;
                    break;
                }
                
                case EnvCommand::STEP: {
                    env_->step(
                        request.actions,
                        response.observations,
                        response.reward,
                        response.terminated
                    );
                    
                    // Auto-reset if environment terminated
                    if (response.terminated) {
                        env_->reset(response.observations);
                        // Keep terminated flag true so caller knows episode ended
                    }
                    break;
                }
                
                case EnvCommand::SHUTDOWN: {
                    running_.store(false);
                    response.success = false;
                    response_channel_.send(std::move(response));
                    return;
                }
            }
            
            // Send response back
            response_channel_.send(std::move(response));
            ready_.store(true);
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
    
    // Create all worker threads
    workers_.reserve(num_envs);
    for (int i = 0; i < num_envs; ++i) {
        workers_.push_back(std::make_unique<EnvWorker>(i));
    }
    
    // Wait for all workers to initialize
    bool all_ready = false;
    while (!all_ready) {
        all_ready = true;
        for (const auto& worker : workers_) {
            if (!worker->is_ready()) {
                all_ready = false;
                break;
            }
        }
        if (!all_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

VecEnv::~VecEnv() {
    // Workers will be shut down by their destructors
}

std::vector<std::array<std::array<float, 138>, 4>> VecEnv::reset() {
    std::vector<std::array<std::array<float, 138>, 4>> all_observations;
    all_observations.reserve(num_envs_);
    
    // Send reset commands to all workers simultaneously (non-blocking)
    for (int i = 0; i < num_envs_; ++i) {
        EnvRequest request;
        request.command = EnvCommand::RESET;
        
        while (!workers_[i]->send_request(std::move(request))) {
            // Spin until we can send (worker is processing previous request)
            std::this_thread::yield();
            request.command = EnvCommand::RESET; // Recreate request if send failed
        }
    }
    
    // Collect responses from all workers
    for (int i = 0; i < num_envs_; ++i) {
        auto response_opt = workers_[i]->recv_response();
        
        if (response_opt.has_value() && response_opt->success) {
            all_observations.push_back(response_opt->observations);
        } else {
            throw std::runtime_error("Worker " + std::to_string(i) + " failed to reset");
        }
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
    
    // Send step commands to all workers simultaneously (non-blocking)
    for (int i = 0; i < num_envs_; ++i) {
        EnvRequest request;
        request.command = EnvCommand::STEP;
        request.actions = actions[i];
        
        while (!workers_[i]->send_request(std::move(request))) {
            // Spin until we can send
            std::this_thread::yield();
            // Recreate request if send failed
            request.command = EnvCommand::STEP;
            request.actions = actions[i];
        }
        
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
    
    // Collect responses from all workers
    for (int i = 0; i < num_envs_; ++i) {
        if (!async_pending_[i]) {
            throw std::runtime_error(
                "step_wait called without corresponding step_async for worker " + 
                std::to_string(i)
            );
        }
        
        auto response_opt = workers_[i]->recv_response();
        
        if (response_opt.has_value() && response_opt->success) {
            all_observations.push_back(response_opt->observations);
            all_rewards.push_back(response_opt->reward);
            all_dones.push_back(response_opt->terminated);
        } else {
            throw std::runtime_error("Worker " + std::to_string(i) + " failed to step");
        }
        
        async_pending_[i] = false;
    }
    
    return std::make_tuple(
        std::move(all_observations),
        std::move(all_rewards),
        std::move(all_dones)
    );
}