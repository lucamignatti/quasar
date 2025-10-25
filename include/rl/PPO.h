#pragma once

#include "rl/MLP.h"
#include <torch/torch.h>
#include <vector>
#include <memory>

namespace rl {

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

// PPO Algorithm Implementation with GPU-optimized buffers
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
        int n_envs = 1;                   // Number of parallel environments
        bool normalize_advantages = true;  // Normalize advantages
        
        // Performance options
        bool use_mixed_precision = false; // FP16 training (ROCm/CUDA)
        bool use_jit = false;             // TorchScript compilation
        bool pin_memory = false;          // Pinned memory for transfers
        
        // Device
        torch::DeviceType device = torch::kCPU;
    };
    
    explicit PPO(const Config& config);
    ~PPO() = default;
    
    // Store a transition in the experience buffer (GPU-resident)
    void store_transition(
        const torch::Tensor& obs,
        const torch::Tensor& action,
        const torch::Tensor& log_prob,
        const torch::Tensor& reward,
        const torch::Tensor& done,
        const torch::Tensor& value
    );
    
    // Compute advantages and returns using vectorized GAE
    void compute_advantages(const torch::Tensor& last_values);
    
    // Update policy using collected experience
    PPOStats update();
    
    // Get action from current policy
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_action(
        const torch::Tensor& obs, 
        bool deterministic = false);
    
    // Get value estimate
    torch::Tensor get_value(const torch::Tensor& obs);
    
    // Get both action and value in one forward pass (optimized)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_action_and_value(
        const torch::Tensor& obs,
        bool deterministic = false);
    
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
    
    // Clear buffer (useful for manual control)
    void clear_buffer();
    
private:
    Config config_;
    MLP policy_;
    torch::optim::Adam optimizer_;
    torch::Device device_;
    
    // GPU-resident preallocated buffers
    torch::Tensor buffer_observations_;   // [n_steps, n_envs, obs_size]
    torch::Tensor buffer_actions_;        // [n_steps, n_envs, action_size]
    torch::Tensor buffer_log_probs_;      // [n_steps, n_envs]
    torch::Tensor buffer_rewards_;        // [n_steps, n_envs]
    torch::Tensor buffer_dones_;          // [n_steps, n_envs]
    torch::Tensor buffer_values_;         // [n_steps, n_envs]
    
    // Computed advantage/return buffers
    torch::Tensor advantages_;            // [n_steps, n_envs]
    torch::Tensor returns_;               // [n_steps, n_envs]
    
    int64_t buffer_pos_;                  // Current position in buffer
    bool buffer_full_;                    // Whether buffer is full
    
    // Preallocate buffers
    void allocate_buffers();
    
    // Compute returns and advantages using vectorized GAE (GPU-optimized)
    void compute_gae_vectorized(const torch::Tensor& last_values);
    
    // Single PPO update epoch with optimizations
    PPOStats update_epoch(
        const torch::Tensor& obs,
        const torch::Tensor& actions,
        const torch::Tensor& old_log_probs,
        const torch::Tensor& advantages,
        const torch::Tensor& returns
    );
    
    // GPU-based minibatch sampling
    std::vector<torch::Tensor> create_minibatch_masks(
        int64_t total_samples, 
        int64_t batch_size
    );
    
    // Compile model with TorchScript (if enabled)
    void compile_model();
    
    // Device detection helper
    static torch::Device detect_device(torch::DeviceType device_type);
};

} // namespace rl