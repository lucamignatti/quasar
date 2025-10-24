#include "vecenv.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>


BatchWorker::BatchWorker(int worker_id, int num_envs_in_batch)
    : worker_id_(worker_id), num_envs_(num_envs_in_batch)
{
    // Pre-allocate all environments
    envs_.reserve(num_envs_);
    for (int i = 0; i < num_envs_; ++i) {
        envs_.push_back(std::make_unique<RLEnv>());
    }

    // Pre-allocate local buffers (Change 3: local buffers to avoid false sharing)
    local_observations_.resize(num_envs_);
    local_rewards_.resize(num_envs_);
    local_dones_.resize(num_envs_);

    // Start persistent worker thread
    thread_ = std::thread(&BatchWorker::worker_loop, this);
}

BatchWorker::~BatchWorker() {
    shutdown();
}

void BatchWorker::shutdown() {
    if (running_.load()) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_.store(false);
            command_ = WorkerCommand::SHUTDOWN;
            work_ready_ = true;
        }
        cv_work_.notify_one();

        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

void BatchWorker::reset_async(std::vector<std::array<std::array<float, 138>, 4>>* observations,
                               int start_idx) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        command_ = WorkerCommand::RESET;
        observations_ = observations;
        start_idx_ = start_idx;
        work_done_ = false;
        work_ready_ = true;
    }
    cv_work_.notify_one();
}

void BatchWorker::step_async(
    const std::vector<std::array<int, 4>>* actions,
    std::vector<std::array<std::array<float, 138>, 4>>* observations,
    std::vector<float>* rewards,
    std::vector<uint8_t>* dones,
    int start_idx)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        command_ = WorkerCommand::STEP;
        actions_ = actions;
        observations_ = observations;
        rewards_ = rewards;
        dones_ = dones;
        start_idx_ = start_idx;
        work_done_ = false;
        work_ready_ = true;
    }
    cv_work_.notify_one();
}

void BatchWorker::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this] { return work_done_; });
}

void BatchWorker::worker_loop() {
    while (running_.load()) {
        WorkerCommand cmd;

        // Wait for work
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_work_.wait(lock, [this] { return work_ready_ || !running_.load(); });

            if (!running_.load()) {
                break;
            }

            cmd = command_;
            work_ready_ = false;
        }

        try {
            switch (cmd) {
                case WorkerCommand::RESET: {
                    for (int i = 0; i < num_envs_; ++i) {
                        envs_[i]->reset((*observations_)[start_idx_ + i]);
                    }
                    break;
                }

                case WorkerCommand::STEP: {
                    // Write to local buffers first (Change 3: avoid false sharing)
                    for (int i = 0; i < num_envs_; ++i) {
                        int idx = start_idx_ + i;
                        bool terminated = false;
                        envs_[i]->step((*actions_)[idx], local_observations_[i],
                                      local_rewards_[i], terminated);
                        local_dones_[i] = terminated ? 1 : 0;

                        // reset on termination
                        if (terminated) {
                            envs_[i]->reset(local_observations_[i]);
                        }
                    }
                    
                    // Copy local results to shared buffers once at the end
                    for (int i = 0; i < num_envs_; ++i) {
                        int idx = start_idx_ + i;
                        (*observations_)[idx] = local_observations_[i];
                        (*rewards_)[idx] = local_rewards_[i];
                        (*dones_)[idx] = local_dones_[i];
                    }
                    break;
                }

                case WorkerCommand::SHUTDOWN:
                    running_.store(false);
                    break;

                case WorkerCommand::IDLE:
                    break;
            }
        } catch (const std::exception& e) {
            std::cerr << "Worker " << worker_id_ << " error: " << e.what() << std::endl;
        }

        // Signal completion
        {
            std::lock_guard<std::mutex> lock(mutex_);
            work_done_ = true;
            command_ = WorkerCommand::IDLE;
        }
        cv_done_.notify_one();
    }
}

VecEnv::VecEnv(int num_envs, int num_threads)
    : num_envs_(num_envs)
{
    if (num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }

    // Default to hardware concurrency if not specified
    if (num_threads <= 0) {
        num_threads_ = std::thread::hardware_concurrency();
        if (num_threads_ == 0) num_threads_ = 4;
    } else {
        num_threads_ = num_threads;
    }

    num_threads_ = std::min(num_threads_, num_envs);

    envs_per_thread_ = (num_envs + num_threads_ - 1) / num_threads_;

    // Change 1: Pre-allocate buffers once
    all_observations_.resize(num_envs_);
    all_rewards_.resize(num_envs_);
    all_dones_.resize(num_envs_);

    // Create workers with persistent threads
    workers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        int start_env = i * envs_per_thread_;
        int end_env = std::min(start_env + envs_per_thread_, num_envs);
        int batch_size = end_env - start_env;

        if (batch_size > 0) {
            workers_.push_back(std::make_unique<BatchWorker>(i, batch_size));
        }
    }
}

VecEnv::~VecEnv() {
    for (auto& worker : workers_) {
        worker->shutdown();
    }
}

std::vector<std::array<std::array<float, 138>, 4>> VecEnv::reset() {
    std::vector<std::array<std::array<float, 138>, 4>> all_observations(num_envs_);

    for (size_t t = 0; t < workers_.size(); ++t) {
        int start_env = t * envs_per_thread_;
        workers_[t]->reset_async(&all_observations, start_env);
    }

    for (auto& worker : workers_) {
        worker->wait();
    }

    return all_observations;
}

std::tuple<
    std::vector<std::array<std::array<float, 138>, 4>>,
    std::vector<float>,
    std::vector<uint8_t>
> VecEnv::step(const std::vector<std::array<int, 4>>& actions) {
    if (actions.size() != static_cast<size_t>(num_envs_)) {
        throw std::invalid_argument(
            "Number of action arrays (" + std::to_string(actions.size()) +
            ") must match num_envs (" + std::to_string(num_envs_) + ")"
        );
    }

    // Change 1: Reuse pre-allocated buffers instead of allocating every call
    // No allocation here - buffers already sized in constructor

    for (size_t t = 0; t < workers_.size(); ++t) {
        int start_env = t * envs_per_thread_;
        workers_[t]->step_async(&actions, &all_observations_, &all_rewards_, &all_dones_, start_env);
    }

    for (auto& worker : workers_) {
        worker->wait();
    }

    return std::make_tuple(
        all_observations_,
        all_rewards_,
        all_dones_
    );
}
