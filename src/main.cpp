#include "env/vecenv.h"
#include "tracing.h"
#include "rl/PPO.h"
#include "rl/MLP.h"
#include "RocketSim.h"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <string>
#include <cstring>
#include <iomanip>

struct TrainingConfig {
    // Environment settings
    int num_envs = 24;
    int num_threads = 0;

    // Training settings
    int max_steps = 1000000;
    int n_steps = 2048;
    int save_interval = 100000;
    std::string save_path = "models/ppo_model.pt";
    std::string load_path = "";

    // PPO hyperparameters
    float learning_rate = 3e-4f;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float clip_epsilon = 0.2f;
    float value_loss_coef = 0.5f;
    float entropy_coef = 0.01f;
    float max_grad_norm = 0.5f;
    int num_epochs = 4;
    int batch_size = 256;

    // Network architecture
    int hidden_size = 256;
    int num_hidden_layers = 2;

    // Misc
    bool use_cuda = false;
    int log_interval = 10;
    bool deterministic = false;

    void print() const {
        std::cout << "\n=== Training Configuration ===\n";
        std::cout << "Environment:\n";
        std::cout << "  num_envs: " << num_envs << "\n";
        std::cout << "  num_threads: " << num_threads << "\n";
        std::cout << "\nTraining:\n";
        std::cout << "  max_steps: " << max_steps << "\n";
        std::cout << "  n_steps: " << n_steps << "\n";
        std::cout << "  save_interval: " << save_interval << "\n";
        std::cout << "  save_path: " << save_path << "\n";
        std::cout << "  load_path: " << (load_path.empty() ? "none" : load_path) << "\n";
        std::cout << "\nPPO Hyperparameters:\n";
        std::cout << "  learning_rate: " << learning_rate << "\n";
        std::cout << "  gamma: " << gamma << "\n";
        std::cout << "  gae_lambda: " << gae_lambda << "\n";
        std::cout << "  clip_epsilon: " << clip_epsilon << "\n";
        std::cout << "  value_loss_coef: " << value_loss_coef << "\n";
        std::cout << "  entropy_coef: " << entropy_coef << "\n";
        std::cout << "  max_grad_norm: " << max_grad_norm << "\n";
        std::cout << "  num_epochs: " << num_epochs << "\n";
        std::cout << "  batch_size: " << batch_size << "\n";
        std::cout << "\nNetwork Architecture:\n";
        std::cout << "  hidden_size: " << hidden_size << "\n";
        std::cout << "  num_hidden_layers: " << num_hidden_layers << "\n";
        std::cout << "\nMisc:\n";
        std::cout << "  use_cuda: " << (use_cuda ? "true" : "false") << "\n";
        std::cout << "  log_interval: " << log_interval << "\n";
        std::cout << "  deterministic: " << (deterministic ? "true" : "false") << "\n";
        std::cout << "==============================\n\n";
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  Environment:\n";
    std::cout << "    --num-envs N              Number of parallel environments (default: 24)\n";
    std::cout << "    --num-threads N           Number of worker threads, 0=auto (default: 0)\n";
    std::cout << "\n  Training:\n";
    std::cout << "    --max-steps N             Maximum training steps (default: 1000000)\n";
    std::cout << "    --n-steps N               Steps per rollout (default: 2048)\n";
    std::cout << "    --save-interval N         Save model every N steps (default: 100000)\n";
    std::cout << "    --save-path PATH          Path to save model (default: models/ppo_model.pt)\n";
    std::cout << "    --load-path PATH          Path to load model from (default: none)\n";
    std::cout << "\n  PPO Hyperparameters:\n";
    std::cout << "    --learning-rate FLOAT     Learning rate (default: 3e-4)\n";
    std::cout << "    --gamma FLOAT             Discount factor (default: 0.99)\n";
    std::cout << "    --gae-lambda FLOAT        GAE lambda parameter (default: 0.95)\n";
    std::cout << "    --clip-epsilon FLOAT      PPO clipping parameter (default: 0.2)\n";
    std::cout << "    --value-loss-coef FLOAT   Value loss coefficient (default: 0.5)\n";
    std::cout << "    --entropy-coef FLOAT      Entropy coefficient (default: 0.01)\n";
    std::cout << "    --max-grad-norm FLOAT     Max gradient norm (default: 0.5)\n";
    std::cout << "    --num-epochs N            PPO epochs per update (default: 4)\n";
    std::cout << "    --batch-size N            Minibatch size (default: 256)\n";
    std::cout << "\n  Network Architecture:\n";
    std::cout << "    --hidden-size N           Hidden layer size (default: 256)\n";
    std::cout << "    --num-hidden-layers N     Number of hidden layers (default: 2)\n";
    std::cout << "\n  Misc:\n";
    std::cout << "    --cuda                    Use CUDA if available (default: CPU)\n";
    std::cout << "    --log-interval N          Log every N updates (default: 10)\n";
    std::cout << "    --deterministic           Use deterministic actions (for evaluation)\n";
    std::cout << "    --help                    Show this help message\n";
    std::cout << std::endl;
}

TrainingConfig parse_args(int argc, char* argv[]) {
    TrainingConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        }
        else if (arg == "--num-envs" && i + 1 < argc) {
            config.num_envs = std::atoi(argv[++i]);
        }
        else if (arg == "--num-threads" && i + 1 < argc) {
            config.num_threads = std::atoi(argv[++i]);
        }
        else if (arg == "--max-steps" && i + 1 < argc) {
            config.max_steps = std::atoi(argv[++i]);
        }
        else if (arg == "--n-steps" && i + 1 < argc) {
            config.n_steps = std::atoi(argv[++i]);
        }
        else if (arg == "--save-interval" && i + 1 < argc) {
            config.save_interval = std::atoi(argv[++i]);
        }
        else if (arg == "--save-path" && i + 1 < argc) {
            config.save_path = argv[++i];
        }
        else if (arg == "--load-path" && i + 1 < argc) {
            config.load_path = argv[++i];
        }
        else if (arg == "--learning-rate" && i + 1 < argc) {
            config.learning_rate = std::atof(argv[++i]);
        }
        else if (arg == "--gamma" && i + 1 < argc) {
            config.gamma = std::atof(argv[++i]);
        }
        else if (arg == "--gae-lambda" && i + 1 < argc) {
            config.gae_lambda = std::atof(argv[++i]);
        }
        else if (arg == "--clip-epsilon" && i + 1 < argc) {
            config.clip_epsilon = std::atof(argv[++i]);
        }
        else if (arg == "--value-loss-coef" && i + 1 < argc) {
            config.value_loss_coef = std::atof(argv[++i]);
        }
        else if (arg == "--entropy-coef" && i + 1 < argc) {
            config.entropy_coef = std::atof(argv[++i]);
        }
        else if (arg == "--max-grad-norm" && i + 1 < argc) {
            config.max_grad_norm = std::atof(argv[++i]);
        }
        else if (arg == "--num-epochs" && i + 1 < argc) {
            config.num_epochs = std::atoi(argv[++i]);
        }
        else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::atoi(argv[++i]);
        }
        else if (arg == "--hidden-size" && i + 1 < argc) {
            config.hidden_size = std::atoi(argv[++i]);
        }
        else if (arg == "--num-hidden-layers" && i + 1 < argc) {
            config.num_hidden_layers = std::atoi(argv[++i]);
        }
        else if (arg == "--cuda") {
            config.use_cuda = true;
        }
        else if (arg == "--log-interval" && i + 1 < argc) {
            config.log_interval = std::atoi(argv[++i]);
        }
        else if (arg == "--deterministic") {
            config.deterministic = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    return config;
}

// Convert vecenv observations to torch tensor
torch::Tensor obs_to_tensor(const std::vector<std::array<std::array<float, 132>, 4>>& obs) {
    int num_envs = obs.size();
    int num_agents = 4;
    int obs_size = 132;

    // For simplicity, we'll just use the first agent's observations
    // In a full multi-agent setup, you'd handle all agents
    auto tensor = torch::zeros({num_envs, obs_size});
    auto accessor = tensor.accessor<float, 2>();

    for (int i = 0; i < num_envs; ++i) {
        for (int j = 0; j < obs_size; ++j) {
            accessor[i][j] = obs[i][0][j];  // Using first agent
        }
    }

    return tensor;
}

// Convert torch tensor actions to vecenv format
std::vector<std::array<int, 4>> tensor_to_actions(const torch::Tensor& actions) {
    auto actions_cpu = actions.to(torch::kCPU);
    auto accessor = actions_cpu.accessor<float, 2>();
    int num_envs = actions_cpu.size(0);

    std::vector<std::array<int, 4>> result(num_envs);

    // Convert continuous actions to discrete actions
    // Actions are [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    for (int i = 0; i < num_envs; ++i) {
        // Simple mapping from continuous to discrete
        // This is a placeholder - you'd want a proper action mapping
        for (int j = 0; j < 4; ++j) {
            if (j < actions_cpu.size(1)) {
                result[i][j] = static_cast<int>(accessor[i][j] * 4.0f + 4.0f);  // Map [-1,1] to [0,8]
                result[i][j] = std::max(0, std::min(8, result[i][j]));
            } else {
                result[i][j] = 0;
            }
        }
    }

    return result;
}

int main(int argc, char* argv[]) {

    tracing::Tracer::Get().Start("perfetto_trace.json");
    TRACE_THREAD_NAME("main");

    // Parse command line arguments
    TrainingConfig config = parse_args(argc, argv);
    config.print();

    {
        TRACE_SCOPE("program");

        // Initialize RocketSim
        {
            TRACE_SCOPE("rocketsim_init");
            std::filesystem::path meshesPath = "collision_meshes";
            if (!std::filesystem::exists(meshesPath)) {
                std::cerr << "ERROR: Meshes path not found. Please run collision_mesh_downloader.py" << std::endl;
                return 1;
            }
            RocketSim::Init(meshesPath, true);
        }

        // Initialize vectorized environment
        VecEnv vecenv(config.num_envs, config.num_threads);

        // Initialize PPO
        rl::PPO::Config ppo_config;
        ppo_config.obs_size = 132;
        ppo_config.action_size = 8;
        ppo_config.hidden_size = config.hidden_size;
        ppo_config.num_hidden_layers = config.num_hidden_layers;
        ppo_config.learning_rate = config.learning_rate;
        ppo_config.gamma = config.gamma;
        ppo_config.gae_lambda = config.gae_lambda;
        ppo_config.clip_epsilon = config.clip_epsilon;
        ppo_config.value_loss_coef = config.value_loss_coef;
        ppo_config.entropy_coef = config.entropy_coef;
        ppo_config.max_grad_norm = config.max_grad_norm;
        ppo_config.num_epochs = config.num_epochs;
        ppo_config.batch_size = config.batch_size;
        ppo_config.n_steps = config.n_steps;

        // Check for CUDA
        if (config.use_cuda && torch::cuda::is_available()) {
            ppo_config.device = torch::kCUDA;
            std::cout << "Using CUDA device" << std::endl;
        } else {
            ppo_config.device = torch::kCPU;
            std::cout << "Using CPU device" << std::endl;
        }

        rl::PPO ppo(ppo_config);

        // Load model if specified
        if (!config.load_path.empty()) {
            try {
                ppo.load(config.load_path);
            } catch (const std::exception& e) {
                std::cerr << "Error loading model: " << e.what() << std::endl;
                std::cerr << "Starting with fresh model..." << std::endl;
            }
        }

        // Create save directory if it doesn't exist
        std::filesystem::path save_dir = std::filesystem::path(config.save_path).parent_path();
        if (!save_dir.empty() && !std::filesystem::exists(save_dir)) {
            std::filesystem::create_directories(save_dir);
        }

        // Training loop
        auto start_time = std::chrono::high_resolution_clock::now();

        auto obs_raw = vecenv.reset();
        auto obs = obs_to_tensor(obs_raw);

        int total_steps = 0;
        int num_updates = 0;
        float total_reward = 0.0f;
        int num_episodes = 0;
        int last_logged_steps = 0;
        auto last_log_time = std::chrono::high_resolution_clock::now();

        std::cout << "\nStarting training...\n" << std::endl;

        {
            TRACE_SCOPE("training_loop");

            while (total_steps < config.max_steps) {
                // Collect rollout
                for (int step = 0; step < config.n_steps; ++step) {
                    // Get actions from policy
                    auto [actions, log_probs, entropy] = ppo.get_action(obs, config.deterministic);
                    auto values = ppo.get_value(obs);

                    // Step environment
                    auto actions_discrete = tensor_to_actions(actions);
                    auto [next_obs_raw, rewards, dones] = vecenv.step(actions_discrete);
                    auto next_obs = obs_to_tensor(next_obs_raw);

                    // Convert rewards and dones to tensors
                    auto rewards_tensor = torch::from_blob(
                        rewards.data(),
                        {static_cast<int64_t>(rewards.size())},
                        torch::kFloat32
                    ).clone();

                    auto dones_tensor = torch::from_blob(
                        dones.data(),
                        {static_cast<int64_t>(dones.size())},
                        torch::kUInt8
                    ).to(torch::kFloat32).clone();

                    // Store transition
                    ppo.store_transition(obs, actions, log_probs, rewards_tensor, dones_tensor, values);

                    // Update stats
                    total_reward += rewards_tensor.sum().item<float>();
                    num_episodes += dones_tensor.sum().item<int>();

                    obs = next_obs;
                    total_steps += config.num_envs;

                    if (total_steps >= config.max_steps) {
                        break;
                    }
                }

                // Only update if buffer is ready
                if (!ppo.is_ready_for_update()) {
                    break;
                }

                // Compute advantages
                auto last_values = ppo.get_value(obs);
                ppo.compute_advantages(last_values);

                // Update policy
                auto stats = ppo.update();
                num_updates++;

                // Logging
                if (num_updates % config.log_interval == 0) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = current_time - start_time;
                    std::chrono::duration<double> elapsed_since_log = current_time - last_log_time;
                    double fps = total_steps / elapsed.count();
                    int steps_since_log = total_steps - last_logged_steps;
                    double sps = steps_since_log / elapsed_since_log.count();

                    std::cout << std::fixed << std::setprecision(2);
                    std::cout << "Update: " << num_updates
                              << " | Steps: " << total_steps
                              << " | FPS: " << fps
                              << " | SPS: " << sps
                              << " | Avg Reward: " << (num_episodes > 0 ? total_reward / num_episodes : 0.0f)
                              << "\n";
                    std::cout << "  Policy Loss: " << stats.policy_loss
                              << " | Value Loss: " << stats.value_loss
                              << " | Entropy: " << stats.entropy
                              << "\n";
                    std::cout << "  Approx KL: " << stats.approx_kl
                              << " | Clip Frac: " << stats.clip_fraction
                              << " | Explained Var: " << stats.explained_variance
                              << std::endl;

                    total_reward = 0.0f;
                    num_episodes = 0;
                    last_logged_steps = total_steps;
                    last_log_time = current_time;
                }

                // Save model
                if (total_steps % config.save_interval == 0) {
                    ppo.save(config.save_path);
                }
            }
        }

        // Final save
        ppo.save(config.save_path);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = end_time - start_time;

        std::cout << "\n=== Training Complete ===\n";
        std::cout << "Total steps: " << total_steps << "\n";
        std::cout << "Total updates: " << num_updates << "\n";
        std::cout << "Total time: " << total_time.count() << " seconds\n";
        std::cout << "Average FPS: " << total_steps / total_time.count() << std::endl;
    }

    tracing::Tracer::Get().Stop();

    return 0;
}
