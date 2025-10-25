#include "rl/PPO.h"
#include <iostream>
#include <algorithm>
#include <random>

namespace rl {

// ExperienceBuffer implementation
void ExperienceBuffer::clear() {
    observations.clear();
    actions.clear();
    log_probs.clear();
    rewards.clear();
    dones.clear();
    values.clear();
}

size_t ExperienceBuffer::size() const {
    return observations.size();
}

bool ExperienceBuffer::is_empty() const {
    return observations.empty();
}

// PPOStats implementation
void PPOStats::reset() {
    policy_loss = 0.0f;
    value_loss = 0.0f;
    entropy = 0.0f;
    approx_kl = 0.0f;
    clip_fraction = 0.0f;
    explained_variance = 0.0f;
    num_updates = 0;
}

void PPOStats::print() const {
    std::cout << "PPO Stats:\n"
              << "  Policy Loss: " << policy_loss << "\n"
              << "  Value Loss: " << value_loss << "\n"
              << "  Entropy: " << entropy << "\n"
              << "  Approx KL: " << approx_kl << "\n"
              << "  Clip Fraction: " << clip_fraction << "\n"
              << "  Explained Variance: " << explained_variance << "\n"
              << "  Num Updates: " << num_updates << std::endl;
}

// PPO implementation
PPO::PPO(const Config& config)
    : config_(config),
      policy_(config.obs_size, config.hidden_size, config.action_size, config.num_hidden_layers),
      optimizer_(policy_->parameters(), torch::optim::AdamOptions(config.learning_rate)),
      device_(config.device) {
    
    policy_->to(device_);
    std::cout << "PPO initialized with device: " << device_ << std::endl;
}

void PPO::store_transition(
    const torch::Tensor& obs,
    const torch::Tensor& action,
    const torch::Tensor& log_prob,
    const torch::Tensor& reward,
    const torch::Tensor& done,
    const torch::Tensor& value) {
    
    buffer_.observations.push_back(obs.to(torch::kCPU));
    buffer_.actions.push_back(action.to(torch::kCPU));
    buffer_.log_probs.push_back(log_prob.to(torch::kCPU));
    buffer_.rewards.push_back(reward.to(torch::kCPU));
    buffer_.dones.push_back(done.to(torch::kCPU));
    buffer_.values.push_back(value.to(torch::kCPU));
}

bool PPO::is_ready_for_update() const {
    return buffer_.size() >= static_cast<size_t>(config_.n_steps);
}

void PPO::compute_gae(const torch::Tensor& last_values) {
    int n_steps = buffer_.size();
    advantages_.clear();
    returns_.clear();
    advantages_.resize(n_steps);
    returns_.resize(n_steps);
    
    // Stack all tensors
    auto rewards = torch::stack(buffer_.rewards);      // [n_steps, n_envs]
    auto dones = torch::stack(buffer_.dones);          // [n_steps, n_envs]
    auto values = torch::stack(buffer_.values);        // [n_steps, n_envs]
    
    // GAE computation
    torch::Tensor gae = torch::zeros_like(last_values);
    
    for (int t = n_steps - 1; t >= 0; --t) {
        torch::Tensor next_values;
        if (t == n_steps - 1) {
            next_values = last_values;
        } else {
            next_values = values[t + 1];
        }
        
        // TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
        auto delta = rewards[t] + config_.gamma * next_values * (1.0f - dones[t]) - values[t];
        
        // GAE: A = delta + gamma * lambda * (1 - done) * A_next
        gae = delta + config_.gamma * config_.gae_lambda * (1.0f - dones[t]) * gae;
        
        advantages_[t] = gae.clone();
        returns_[t] = advantages_[t] + values[t];
    }
}

void PPO::compute_advantages(const torch::Tensor& last_values) {
    compute_gae(last_values);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PPO::get_action(
    const torch::Tensor& obs, bool deterministic) {
    
    auto obs_device = obs.to(device_);
    return policy_->get_action(obs_device, deterministic);
}

torch::Tensor PPO::get_value(const torch::Tensor& obs) {
    auto obs_device = obs.to(device_);
    return policy_->get_value(obs_device);
}

std::vector<std::vector<int64_t>> PPO::create_minibatch_indices(
    int64_t total_samples, int64_t batch_size) {
    
    // Create shuffled indices
    std::vector<int64_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Split into minibatches
    std::vector<std::vector<int64_t>> minibatches;
    for (int64_t i = 0; i < total_samples; i += batch_size) {
        int64_t end = std::min(i + batch_size, total_samples);
        std::vector<int64_t> batch(indices.begin() + i, indices.begin() + end);
        minibatches.push_back(batch);
    }
    
    return minibatches;
}

PPOStats PPO::update_epoch(
    const torch::Tensor& obs,
    const torch::Tensor& actions,
    const torch::Tensor& old_log_probs,
    const torch::Tensor& advantages,
    const torch::Tensor& returns) {
    
    PPOStats stats;
    int64_t total_samples = obs.size(0);
    
    auto minibatch_indices = create_minibatch_indices(total_samples, config_.batch_size);
    
    for (const auto& indices : minibatch_indices) {
        // Create index tensor
        auto idx_tensor = torch::tensor(indices, torch::kLong);
        
        // Get minibatch data
        auto mb_obs = obs.index_select(0, idx_tensor);
        auto mb_actions = actions.index_select(0, idx_tensor);
        auto mb_old_log_probs = old_log_probs.index_select(0, idx_tensor);
        auto mb_advantages = advantages.index_select(0, idx_tensor);
        auto mb_returns = returns.index_select(0, idx_tensor);
        
        // Normalize advantages (within minibatch)
        if (config_.normalize_advantages) {
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8);
        }
        
        // Evaluate actions with current policy
        auto [log_probs, entropy, values] = policy_->evaluate_actions(mb_obs, mb_actions);
        
        // Compute ratio (pi_current / pi_old)
        auto ratio = torch::exp(log_probs - mb_old_log_probs);
        
        // Compute surrogate losses
        auto surr1 = ratio * mb_advantages;
        auto surr2 = torch::clamp(ratio, 1.0f - config_.clip_epsilon, 1.0f + config_.clip_epsilon) * mb_advantages;
        auto policy_loss = -torch::min(surr1, surr2).mean();
        
        // Value loss (clipped)
        auto value_loss = torch::nn::functional::mse_loss(values, mb_returns);
        
        // Entropy bonus
        auto entropy_loss = -entropy.mean();
        
        // Total loss
        auto loss = policy_loss + config_.value_loss_coef * value_loss + config_.entropy_coef * entropy_loss;
        
        // Optimize
        optimizer_.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(policy_->parameters(), config_.max_grad_norm);
        optimizer_.step();
        
        // Update stats
        stats.policy_loss += policy_loss.item<float>();
        stats.value_loss += value_loss.item<float>();
        stats.entropy += entropy.mean().item<float>();
        
        // Compute approximate KL divergence
        auto approx_kl = ((ratio - 1.0f) - torch::log(ratio)).mean();
        stats.approx_kl += approx_kl.item<float>();
        
        // Compute clip fraction
        auto clip_fraction = ((ratio - 1.0f).abs() > config_.clip_epsilon).to(torch::kFloat32).mean();
        stats.clip_fraction += clip_fraction.item<float>();
        
        stats.num_updates++;
    }
    
    // Average the stats
    if (stats.num_updates > 0) {
        stats.policy_loss /= stats.num_updates;
        stats.value_loss /= stats.num_updates;
        stats.entropy /= stats.num_updates;
        stats.approx_kl /= stats.num_updates;
        stats.clip_fraction /= stats.num_updates;
    }
    
    return stats;
}

PPOStats PPO::update() {
    if (!is_ready_for_update()) {
        throw std::runtime_error("Buffer not ready for update");
    }
    
    // Stack all data
    auto obs = torch::stack(buffer_.observations).to(device_);          // [n_steps, n_envs, obs_size]
    auto actions = torch::stack(buffer_.actions).to(device_);           // [n_steps, n_envs, action_size]
    auto old_log_probs = torch::stack(buffer_.log_probs).to(device_);   // [n_steps, n_envs]
    auto advantages = torch::stack(advantages_).to(device_);             // [n_steps, n_envs]
    auto returns = torch::stack(returns_).to(device_);                   // [n_steps, n_envs]
    
    // Flatten batch dimensions: [n_steps, n_envs, ...] -> [n_steps * n_envs, ...]
    obs = obs.flatten(0, 1);
    actions = actions.flatten(0, 1);
    old_log_probs = old_log_probs.flatten(0, 1);
    advantages = advantages.flatten(0, 1);
    returns = returns.flatten(0, 1);
    
    // Compute explained variance
    auto var_y = returns.var().item<float>();
    auto var_pred = (returns - policy_->get_value(obs)).var().item<float>();
    float explained_var = (var_y > 0) ? (1.0f - var_pred / var_y) : 0.0f;
    
    // Run multiple PPO epochs
    PPOStats total_stats;
    total_stats.reset();
    
    policy_->train();
    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        auto epoch_stats = update_epoch(obs, actions, old_log_probs, advantages, returns);
        
        total_stats.policy_loss += epoch_stats.policy_loss;
        total_stats.value_loss += epoch_stats.value_loss;
        total_stats.entropy += epoch_stats.entropy;
        total_stats.approx_kl += epoch_stats.approx_kl;
        total_stats.clip_fraction += epoch_stats.clip_fraction;
        total_stats.num_updates += epoch_stats.num_updates;
    }
    
    // Average over epochs
    total_stats.policy_loss /= config_.num_epochs;
    total_stats.value_loss /= config_.num_epochs;
    total_stats.entropy /= config_.num_epochs;
    total_stats.approx_kl /= config_.num_epochs;
    total_stats.clip_fraction /= config_.num_epochs;
    total_stats.explained_variance = explained_var;
    
    // Clear buffer and computed values
    buffer_.clear();
    advantages_.clear();
    returns_.clear();
    
    policy_->eval();
    
    return total_stats;
}

void PPO::save(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    policy_->save(archive);
    archive.save_to(path);
    std::cout << "Model saved to: " << path << std::endl;
}

void PPO::load(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    policy_->load(archive);
    policy_->to(device_);
    std::cout << "Model loaded from: " << path << std::endl;
}

float PPO::get_learning_rate() const {
    return optimizer_.param_groups()[0].options().get_lr();
}

void PPO::set_learning_rate(float lr) {
    for (auto& param_group : optimizer_.param_groups()) {
        param_group.options().set_lr(lr);
    }
}

} // namespace rl