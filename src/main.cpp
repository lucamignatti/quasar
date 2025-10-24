#include "vecenv.h"
#include <chrono>

int main() {

    int num_envs = 8;
    int max_steps = 10000;

    VecEnv vecenv(num_envs);
    
    std::vector<std::array<int, 4>> actions(num_envs);
    for (int i = 0; i < num_envs; i++) {
        for (int j = 0; j < 4; j++) {
            actions[i][j] = 0;
        }
    }

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < max_steps; step++) {
        auto [obs, rewards, dones] = vecenv.step(actions);
    }

    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> double_seconds = stop - start;

    int total_steps = max_steps * num_envs;

    std::cout << "Simulated " << total_steps << " steps in " << double_seconds.count() << " seconds, averaging " << total_steps/double_seconds.count() << " steps per second." << std::endl;


}