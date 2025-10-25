#pragma once

#include "rl/MLP.h"
#include <torch/torch.h>
#include <vector>
#include <memory>

namespace rl {

// Experience buffer for storing rollout data
struct ExperienceBuffer {
    std::vector<torch::Tensor> observations;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> log_probs;
    std::vector<torch::Tensor> rewards;
    std::vector<torch::Tensor> dones;
    std::vector<torch::Tensor> values;
    
    void clear();
    size_t size() const;
    bool is_empty() const;
};

// PPO training statistics
struct PPOStats {
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy = 0.0f;
    float approx_kl = 0.0f;
    float clip_fraction = 0.0f;
    float explained_variance = 0.0f;
    int num_updates = 0;
    
    void reset();
    void print() const;
};

// PPO Algorithm Implementation
class PPO {
public:
    struct Config {
        // Network architecture
        int64_t obs_size = 132;
        int64_t action_size = 8;
        int64_t hidden_size = 256;
        int64_t num_hidden_layers = 2;
        
        // PPO hyperparameters
        float learning_rate = 3e-4f;
        float gamma = 0.99f;              // Discount factor
        float gae_lambda = 0.95f;         // GAE parameter
        float clip_epsilon = 0.2f;        // PPO clipping parameter
        float value_loss_coef = 0.5f;     // Value loss coefficient
        float entropy_coef = 0.01f;       // Entropy bonus coefficient
        float max_grad_norm = 0.5f;       // Gradient clipping
        
        // Training parameters
        int num_epochs = 4;               // PPO epochs per update
        int batch_size = 256;             // Minibatch size
        int n_steps = 2048;               // Steps per rollout
        bool normalize_advantages = true;  // Normalize advantages
        
        // Device
        torch::DeviceType device = torch::kCPU;
    };
    
    explicit PPO(const Config& config);
    ~PPO() = default;
    
    // Store a transition in the experience buffer
    void store_transition(
        const torch::Tensor& obs,
        const torch::Tensor& action,
        const torch::Tensor& log_prob,
        const torch::Tensor& reward,
        const torch::Tensor& done,
        const torch::Tensor& value
    );
    
    // Compute advantages and returns using GAE
    void compute_advantages(const torch::Tensor& last_values);
    
    // Update policy using collected experience
    PPOStats update();
    
    // Get action from current policy
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_action(
        const torch::Tensor& obs, 
        bool deterministic = false);
    
    // Get value estimate
    torch::Tensor get_value(const torch::Tensor& obs);
    
    // Check if buffer is ready for update
    bool is_ready_for_update() const;
    
    // Get the policy network
    MLP& get_policy() { return policy_; }
    const MLP& get_policy() const { return policy_; }
    
    // Save/load model
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    // Get configuration
    const Config& get_config() const { return config_; }
    
    // Get current learning rate
    float get_learning_rate() const;
    
    // Set learning rate (for LR scheduling)
    void set_learning_rate(float lr);
    
private:
    Config config_;
    MLP policy_;
    torch::optim::Adam optimizer_;
    torch::Device device_;
    
    ExperienceBuffer buffer_;
    std::vector<torch::Tensor> advantages_;
    std::vector<torch::Tensor> returns_;
    
    // Compute returns and advantages using GAE
    void compute_gae(const torch::Tensor& last_values);
    
    // Single PPO update epoch
    PPOStats update_epoch(
        const torch::Tensor& obs,
        const torch::Tensor& actions,
        const torch::Tensor& old_log_probs,
        const torch::Tensor& advantages,
        const torch::Tensor& returns
    );
    
    // Helper to create minibatches
    std::vector<std::vector<int64_t>> create_minibatch_indices(
        int64_t total_samples, 
        int64_t batch_size
    );
};

} // namespace rl