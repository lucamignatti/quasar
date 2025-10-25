#include "env/vecenv.h"
#include "tracing.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <string>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#elif __APPLE__
#include <pthread.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#endif


BatchWorker::BatchWorker(int worker_id, int num_envs_in_batch)
    : worker_id_(worker_id), num_envs_(num_envs_in_batch)
{
    envs_.reserve(num_envs_);
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
            worker_data_.command = WorkerCommand::SHUTDOWN;
            worker_data_.work_ready = true;
        }
        cv_work_.notify_one();

        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

void BatchWorker::reset_async(std::vector<std::array<std::array<float, 132>, 4>>* observations,
                               int start_idx) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_data_.command = WorkerCommand::RESET;
        observations_ = observations;
        start_idx_ = start_idx;
        worker_data_.work_done = false;
        worker_data_.work_ready = true;
    }
    cv_work_.notify_one();
}

void BatchWorker::step_async(
    const std::vector<std::array<int, 4>>* actions,
    std::vector<std::array<std::array<float, 132>, 4>>* observations,
    std::vector<float>* rewards,
    std::vector<uint8_t>* dones,
    int start_idx)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_data_.command = WorkerCommand::STEP;
        actions_ = actions;
        observations_ = observations;
        rewards_ = rewards;
        dones_ = dones;
        start_idx_ = start_idx;
        worker_data_.work_done = false;
        worker_data_.work_ready = true;
    }
    cv_work_.notify_one();
}

void BatchWorker::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this] { return worker_data_.work_done; });
}

void BatchWorker::worker_loop() {
    std::string thread_name = "worker/" + std::to_string(worker_id_);
    TRACE_THREAD_NAME(thread_name);

    // Pin thread to specific CPU core
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_id_, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#elif __APPLE__
    thread_affinity_policy_data_t policy = { worker_id_ };
    thread_policy_set(pthread_mach_thread_np(pthread_self()),
                     THREAD_AFFINITY_POLICY,
                     (thread_policy_t)&policy, 1);
#endif

    {
        TRACE_SCOPE("worker_env_init");
        for (int i = 0; i < num_envs_; ++i) {
            envs_.push_back(std::make_unique<RLEnv>());
        }
    }

    while (running_.load()) {
        WorkerCommand cmd;

        // Wait for work
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_work_.wait(lock, [this] { return worker_data_.work_ready || !running_.load(); });

            if (!running_.load()) {
                break;
            }

            cmd = worker_data_.command;
            worker_data_.work_ready = false;
        }

        try {
            switch (cmd) {
                case WorkerCommand::RESET: {
                    TRACE_SCOPE("worker_reset");
                    for (int i = 0; i < num_envs_; ++i) {
                        envs_[i]->reset((*observations_)[start_idx_ + i]);
                    }
                    break;
                }

                case WorkerCommand::STEP: {
                    TRACE_SCOPE("worker_step");
                    // Write directly to shared buffers (cache-aligned, no false sharing)
                    for (int i = 0; i < num_envs_; ++i) {
                        int idx = start_idx_ + i;
                        bool terminated = false;
                        envs_[i]->step((*actions_)[idx], (*observations_)[idx],
                                      (*rewards_)[idx], terminated);
                        (*dones_)[idx] = terminated ? 1 : 0;

                        // reset on termination
                        if (terminated) {
                            envs_[i]->reset((*observations_)[idx]);
                        }
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
            worker_data_.work_done = true;
            worker_data_.command = WorkerCommand::IDLE;
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

    // Allocate with aligned allocator
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

std::vector<std::array<std::array<float, 132>, 4>> VecEnv::reset() {
    std::vector<std::array<std::array<float, 132>, 4>> all_observations(num_envs_);

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
    std::vector<std::array<std::array<float, 132>, 4>>,
    std::vector<float>,
    std::vector<uint8_t>
> VecEnv::step(const std::vector<std::array<int, 4>>& actions) {
    if (actions.size() != static_cast<size_t>(num_envs_)) {
        throw std::invalid_argument(
            "Number of action arrays (" + std::to_string(actions.size()) +
            ") must match num_envs (" + std::to_string(num_envs_) + ")"
        );
    }

    // Dispatch all work at once with minimal locking
    for (size_t t = 0; t < workers_.size(); ++t) {
        int start_env = t * envs_per_thread_;
        workers_[t]->step_async(&actions, &all_observations_, &all_rewards_, &all_dones_, start_env);
    }

    // Wait for all workers to complete
    for (auto& worker : workers_) {
        worker->wait();
    }

    return std::make_tuple(
        all_observations_,
        all_rewards_,
        all_dones_
    );
}
