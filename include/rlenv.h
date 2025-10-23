#ifndef RLENV_H
#define RLENV_H

#include <array>
#include "Sim/Arena/Arena.h"
#include <vector>

class RLEnv {
    public:
        RLEnv();
        ~RLEnv();

        RLEnv(const RLEnv&) = delete;
        RLEnv& operator=(const RLEnv&) = delete;

        void reset(std::array<std::array<float, 138>, 4> &obs);
        void step(const std::array<int, 4>& actions, std::array<std::array<float, 138>, 4> &obs, float &reward, bool &terminated);
    private:

        RocketSim::Arena* arena;
        std::array<RocketSim::Car*, 4> cars;

        std::mt19937 gen;
        std::uniform_int_distribution<> kickoffDist;

        void _resetToKickoff();

        std::array<std::array<float, 138>, 4> _getObs();

        std::array<float, 20> _generateCarObs(RocketSim::Car* car, bool inverted) const;

        void _buildLookupTable();

        std::vector<std::array<float, 8>> m_lookupTable;

        static constexpr int ACTION_REPEATS = 8;

        static constexpr float POS_COEF = 1.0f / 2300.0f;
        static constexpr float ANG_COEF = 1.0f / M_PI;
        static constexpr float LIN_VEL_COEF = 1.0f / 2300.0f;
        static constexpr float ANG_VEL_COEF = 1.0f / M_PI;
        static constexpr float PAD_TIMER_COEF = 1.0f / 10.0f;
        static constexpr float BOOST_COEF = 1.0f / 100.0f;

        static constexpr int ZERO_PADDING = 2; // 2v2
        static constexpr int NUM_AGENTS = 4;
        static constexpr int CAR_OBS_SIZE = 20;
        static constexpr int BOOST_PAD_COUNT = 34;
        static constexpr int OBS_SIZE = 138;
};




#endif
