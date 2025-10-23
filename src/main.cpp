#include "rlenv.h"
#include <chrono>

int main() {

    int episodes = 100;

    RLEnv env = RLEnv();
    std::array<std::array<float, 138>, 4> obs;
    float reward = 0;
    std::array<int, 4> actions = {};

    int steps = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < episodes; i++) {
        env.reset(obs);
        bool terminated = false;
        while (!terminated) {

            env.step(actions, obs, reward, terminated);

            ++steps;

        }
    }

    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> double_seconds = stop - start;

    std::cout << "Simulated " << steps << " steps in " << double_seconds.count() << " seconds, averaging " << steps/double_seconds.count() << " steps per second." << std::endl;


}
