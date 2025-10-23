#include "rlenv.h"

int main() {

    int episodes = 100;

    RLEnv env = RLEnv();
    std::array<std::array<float, 138>, 4> obs;
    float reward = 0;
    std::array<int, 4> actions = {};

    for (int i = 0; i < episodes; i++) {
        env.reset(obs);
        bool terminated = false;
        while (!terminated) {

            env.step(actions, obs, reward, terminated);

        }
    }

}
