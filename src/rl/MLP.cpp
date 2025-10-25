#include "rl/MLP.h"
#include <torch/torch.h>

namespace rl {

MLPImpl::MLPImpl(int64_t input_size, 
                 int64_t hidden_size, 
                 int64_t action_size,
                 int64_t num_hidden_layers)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      action_size_(action_size),
      num_hidden_layers_(num_hidden_layers) {
    
    build_networks();
}

void MLPImpl::build_networks() {
    // Build shared feature extractor
    shared_net_ = torch::nn::Sequential();
    
    // First layer
    shared_net_->push_back(torch::nn::Linear(input_size_, hidden_size_));
    shared_net_->push_back(torch::nn::ReLU());
    
    // Additional hidden layers
    for (int64_t i = 1; i < num_hidden_layers_; ++i) {
        shared_net_->push_back(torch::nn::Linear(hidden_size_, hidden_size_));
        shared_net_->push_back(torch::nn::ReLU());
    }
    
    register_module("shared_net", shared_net_);
    
    // Build actor (policy) head - outputs mean actions
    actor_mean_ = torch::nn::Sequential(
        torch::nn::Linear(hidden_size_, action_size_),
        torch::nn::Tanh()  // Actions in [-1, 1]
    );
    register_module("actor_mean", actor_mean_);
    
    // Learnable log standard deviation for action distribution
    action_log_std_ = register_parameter("action_log_std", 
        torch::zeros({action_size_}, torch::kFloat32));
    
    // Build critic (value) head
    critic_ = torch::nn::Sequential(
        torch::nn::Linear(hidden_size_, 1)
    );
    register_module("critic", critic_);
    
    // Initialize weights with orthogonal initialization
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
            torch::nn::init::constant_(linear->bias, 0.0);
        }
    }
    
    // Special initialization for actor and critic output layers
    auto actor_modules = actor_mean_->named_children();
    for (auto& pair : actor_modules) {
        if (auto* linear = pair.value()->as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(linear->weight, 0.01);
        }
    }
    
    auto critic_modules = critic_->named_children();
    for (auto& pair : critic_modules) {
        if (auto* linear = pair.value()->as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(linear->weight, 1.0);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> MLPImpl::forward_actor(torch::Tensor x) {
    auto features = shared_net_->forward(x);
    auto action_mean = actor_mean_->forward(features);
    auto action_log_std = action_log_std_.expand_as(action_mean);
    return std::make_tuple(action_mean, action_log_std);
}

torch::Tensor MLPImpl::forward_critic(torch::Tensor x) {
    auto features = shared_net_->forward(x);
    return critic_->forward(features);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MLPImpl::forward_actor_critic(torch::Tensor x) {
    // Optimized: compute shared features once
    auto features = shared_net_->forward(x);
    
    // Actor head
    auto action_mean = actor_mean_->forward(features);
    auto action_log_std = action_log_std_.expand_as(action_mean);
    
    // Critic head
    auto values = critic_->forward(features);
    
    return std::make_tuple(action_mean, action_log_std, values);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MLPImpl::get_action(
    torch::Tensor obs, bool deterministic) {
    
    torch::InferenceMode guard;
    
    auto [action_mean, action_log_std] = forward_actor(obs);
    
    if (deterministic) {
        // Return mean action for evaluation
        auto log_probs = torch::zeros({obs.size(0)}, obs.options());
        auto entropy = torch::zeros({obs.size(0)}, obs.options());
        return std::make_tuple(action_mean, log_probs, entropy);
    }
    
    // Sample from Gaussian distribution
    auto action_std = torch::exp(action_log_std);
    auto normal = torch::randn_like(action_mean);
    auto actions = action_mean + action_std * normal;
    
    // Compute log probabilities using the formula for Gaussian distribution
    // log p(x) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
    auto log_probs = -0.5 * torch::pow((actions - action_mean) / action_std, 2) 
                     - action_log_std 
                     - 0.5 * std::log(2.0 * M_PI);
    log_probs = log_probs.sum(-1);
    
    // Compute entropy: H = 0.5 * log(2 * pi * e * sigma^2)
    auto entropy = 0.5 * (std::log(2.0 * M_PI * M_E) + 2.0 * action_log_std);
    entropy = entropy.sum(-1);
    
    return std::make_tuple(actions, log_probs, entropy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MLPImpl::get_action_and_value(
    torch::Tensor obs, bool deterministic) {
    
    torch::InferenceMode guard;
    
    auto [action_mean, action_log_std, values] = forward_actor_critic(obs);
    
    if (deterministic) {
        // Return mean action for evaluation
        auto log_probs = torch::zeros({obs.size(0)}, obs.options());
        auto entropy = torch::zeros({obs.size(0)}, obs.options());
        return std::make_tuple(action_mean, log_probs, entropy, values.squeeze(-1));
    }
    
    // Sample from Gaussian distribution
    auto action_std = torch::exp(action_log_std);
    auto normal = torch::randn_like(action_mean);
    auto actions = action_mean + action_std * normal;
    
    // Compute log probabilities
    auto log_probs = -0.5 * torch::pow((actions - action_mean) / action_std, 2) 
                     - action_log_std 
                     - 0.5 * std::log(2.0 * M_PI);
    log_probs = log_probs.sum(-1);
    
    // Compute entropy
    auto entropy = 0.5 * (std::log(2.0 * M_PI * M_E) + 2.0 * action_log_std);
    entropy = entropy.sum(-1);
    
    return std::make_tuple(actions, log_probs, entropy, values.squeeze(-1));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MLPImpl::evaluate_actions(
    torch::Tensor obs, torch::Tensor actions) {
    
    auto [action_mean, action_log_std, values] = forward_actor_critic(obs);
    
    // Compute log probabilities for given actions
    auto action_std = torch::exp(action_log_std);
    auto log_probs = -0.5 * torch::pow((actions - action_mean) / action_std, 2)
                     - action_log_std 
                     - 0.5 * std::log(2.0 * M_PI);
    log_probs = log_probs.sum(-1);
    
    // Compute entropy
    auto entropy = 0.5 * (std::log(2.0 * M_PI * M_E) + 2.0 * action_log_std);
    entropy = entropy.sum(-1);
    
    return std::make_tuple(log_probs, entropy, values.squeeze(-1));
}

torch::Tensor MLPImpl::get_value(torch::Tensor obs) {
    torch::InferenceMode guard;
    return forward_critic(obs).squeeze(-1);
}

} // namespace rl