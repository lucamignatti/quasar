#pragma once

#include <torch/torch.h>
#include <vector>

namespace rl {

// Multi-Layer Perceptron for actor-critic architecture
class MLPImpl : public torch::nn::Module {
public:
    MLPImpl(int64_t input_size, 
            int64_t hidden_size, 
            int64_t action_size,
            int64_t num_hidden_layers = 2);

    // Forward pass for actor (policy network)
    // Returns: tuple of (action_logits, action_log_std)
    std::tuple<torch::Tensor, torch::Tensor> forward_actor(torch::Tensor x);

    // Forward pass for critic (value network)
    // Returns: state value
    torch::Tensor forward_critic(torch::Tensor x);

    // Get action from policy network
    // Returns: tuple of (actions, log_probs, entropy)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_action(torch::Tensor obs, bool deterministic = false);

    // Evaluate actions (for PPO update)
    // Returns: tuple of (log_probs, entropy, values)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate_actions(
        torch::Tensor obs, 
        torch::Tensor actions);

    // Get the value estimate for observations
    torch::Tensor get_value(torch::Tensor obs);

private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t action_size_;
    int64_t num_hidden_layers_;

    // Shared feature extractor
    torch::nn::Sequential shared_net_;
    
    // Actor (policy) head
    torch::nn::Sequential actor_mean_;
    torch::Tensor action_log_std_;
    
    // Critic (value) head
    torch::nn::Sequential critic_;

    void build_networks();
};

TORCH_MODULE(MLP);

} // namespace rl