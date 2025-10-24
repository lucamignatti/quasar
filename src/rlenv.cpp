#include "rlenv.h"
#include "Sim/Arena/Arena.h"
#include "Math/MathTypes/MathTypes.h"
#include "RocketSim.h"
#include "tracing.h"
#include <algorithm>
#include <vector>

RLEnv::RLEnv()
    : gen(std::random_device{}()),
      kickoffDist(0, 2)
{
    // RocketSim::Init is now called once in main.cpp before VecEnv creation
    arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);

    for (int i = 0; i < 4; i++) {
        if (i % 2 == 0) {
            cars[i] = arena->AddCar(RocketSim::Team::BLUE);
        } else {
            cars[i] = arena->AddCar(RocketSim::Team::ORANGE);
        }
    }

    _resetToKickoff();
    _buildLookupTable();
}

RLEnv::~RLEnv()
{
    delete this->arena;
    this->arena = nullptr;
}

void RLEnv::_resetToKickoff() {

    RocketSim::BallState ballState = {};
    ballState.pos = { 0.f, 0.f, 92.75f };
    arena->ball->SetState(ballState);

    using KickoffPos = std::pair<RocketSim::Vec, float>;
    KickoffPos blue_diag_left   = { { -2048.f, -2560.f, 17.f }, 0.25f * M_PI };
    KickoffPos blue_diag_right  = { { 2048.f, -2560.f, 17.f }, 0.75f * M_PI };
    KickoffPos blue_back_left   = { { -256.f, -3840.f, 17.f }, 0.5f * M_PI };
    KickoffPos blue_back_right  = { { 256.f, -3840.f, 17.f }, 0.5f * M_PI };
    KickoffPos blue_back_center = { { 0.f, -4608.f, 17.f }, 0.5f * M_PI };

    std::pair<KickoffPos, KickoffPos> scenarios[3];
    scenarios[0] = { blue_diag_left, blue_back_center };
    scenarios[1] = { blue_diag_right, blue_back_center };
    scenarios[2] = { blue_back_left, blue_back_right };

    int scenarioIdx = kickoffDist(gen);
    auto bluePair1 = scenarios[scenarioIdx].first;
    auto bluePair2 = scenarios[scenarioIdx].second;

    float defaultBoost = 100.f / 3.f;
    RocketSim::CarState kickoffStates[4];

    kickoffStates[0].pos = bluePair1.first;
    kickoffStates[0].boost = defaultBoost;
    kickoffStates[0].rotMat = RocketSim::Angle(bluePair1.second, 0, 0).ToRotMat();

    kickoffStates[1].pos = { -bluePair1.first.x, -bluePair1.first.y, bluePair1.first.z };
    kickoffStates[1].boost = defaultBoost;
    kickoffStates[1].rotMat = RocketSim::Angle(bluePair1.second - M_PI, 0, 0).ToRotMat();

    kickoffStates[2].pos = bluePair2.first;
    kickoffStates[2].boost = defaultBoost;
    kickoffStates[2].rotMat = RocketSim::Angle(bluePair2.second, 0, 0).ToRotMat();

    kickoffStates[3].pos = { -bluePair2.first.x, -bluePair2.first.y, bluePair2.first.z };
    kickoffStates[3].boost = defaultBoost;
    kickoffStates[3].rotMat = RocketSim::Angle(bluePair2.second - M_PI, 0, 0).ToRotMat();

    for (int i = 0; i < 4; i++) {
        cars[i]->SetState(kickoffStates[i]);
    }
}

void RLEnv::_buildLookupTable() {
    m_lookupTable.clear();

    // Ground
    for (float throttle : {-1.f, 0.f, 1.f}) {
        for (float steer : {-1.f, 0.f, 1.f}) {
            for (float boost : {0.f, 1.f}) {
                for (float handbrake : {0.f, 1.f}) {
                    if (boost == 1.f && throttle != 1.f) continue;
                    // {throttle, steer, pitch, yaw, roll, jump, boost, handbrake}
                    m_lookupTable.push_back({throttle > 0 ? throttle : boost, steer, 0.f, steer, 0.f, 0.f, boost, handbrake});
                }
            }
        }
    }

    // Aerial
    for (float pitch : {-1.f, 0.f, 1.f}) {
        for (float yaw : {-1.f, 0.f, 1.f}) {
            for (float roll : {-1.f, 0.f, 1.f}) {
                for (float jump : {0.f, 1.f}) {
                    for (float boost : {0.f, 1.f}) {
                        if (jump == 1.f && yaw != 0.f) continue; // Only need roll for sideflip
                        if (pitch == 0.f && roll == 0.f && jump == 0.f) continue; // Duplicate with ground

                        float handbrake = (jump == 1.f && (pitch != 0.f || yaw != 0.f || roll != 0.f));
                        // {throttle, steer, pitch, yaw, roll, jump, boost, handbrake}
                        m_lookupTable.push_back({boost, yaw, pitch, yaw, roll, jump, boost, handbrake});
                    }
                }
            }
        }
    }
}

std::array<float, RLEnv::CAR_OBS_SIZE> RLEnv::_generateCarObs(RocketSim::Car* car, bool inverted) const {
    std::array<float, CAR_OBS_SIZE> carObs;
    RocketSim::CarState state = car->GetState();

    RocketSim::Vec forward = state.rotMat.forward;
    RocketSim::Vec up = state.rotMat.up;

    if (inverted) {
        state.pos.x *= -1;
        state.pos.y *= -1;
        state.vel.x *= -1;
        state.vel.y *= -1;
        state.angVel.x *= -1;
        state.angVel.y *= -1;

        forward.x *= -1;
        forward.y *= -1;
        up.x *= -1;
        up.y *= -1;
    }

    int i = 0;
    // Position
    carObs[i++] = state.pos.x * POS_COEF;
    carObs[i++] = state.pos.y * POS_COEF;
    carObs[i++] = state.pos.z * POS_COEF;

    // Orientation
    carObs[i++] = forward.x;
    carObs[i++] = forward.y;
    carObs[i++] = forward.z;
    carObs[i++] = up.x;
    carObs[i++] = up.y;
    carObs[i++] = up.z;

    // Linear Velocity
    carObs[i++] = state.vel.x * LIN_VEL_COEF;
    carObs[i++] = state.vel.y * LIN_VEL_COEF;
    carObs[i++] = state.vel.z * LIN_VEL_COEF;

    // Angular Velocity
    carObs[i++] = state.angVel.x * ANG_VEL_COEF;
    carObs[i++] = state.angVel.y * ANG_VEL_COEF;
    carObs[i++] = state.angVel.z * ANG_VEL_COEF;

    // Other
    carObs[i++] = state.boost * BOOST_COEF;
    carObs[i++] = state.demoRespawnTimer;
    carObs[i++] = static_cast<float>(state.isOnGround);
    carObs[i++] = static_cast<float>(state.isBoosting);
    carObs[i++] = static_cast<float>(state.isSupersonic);

    return carObs;
}

std::array<std::array<float, RLEnv::OBS_SIZE>, RLEnv::NUM_AGENTS> RLEnv::_getObs() {

    std::array<std::array<float, OBS_SIZE>, NUM_AGENTS> allObs;
    auto boostPads = arena->GetBoostPads();

    for (int i = 0; i < NUM_AGENTS; ++i) {
        RocketSim::Car* car = cars[i];
        RocketSim::CarState carState = car->GetState();

        bool inverted = (car->team == RocketSim::Team::ORANGE);

        std::array<float, OBS_SIZE>& obs = allObs[i];

        std::fill(obs.begin(), obs.end(), 0.0f);

        int idx = 0;

        // Ball Info (9 floats)
        RocketSim::BallState ballState = arena->ball->GetState();
        if (inverted) {
            ballState.pos.x *= -1;
            ballState.pos.y *= -1;
            ballState.vel.x *= -1;
            ballState.vel.y *= -1;
            ballState.angVel.x *= -1;
            ballState.angVel.y *= -1;
        }
        obs[idx++] = ballState.pos.x * POS_COEF;
        obs[idx++] = ballState.pos.y * POS_COEF;
        obs[idx++] = ballState.pos.z * POS_COEF;
        obs[idx++] = ballState.vel.x * LIN_VEL_COEF;
        obs[idx++] = ballState.vel.y * LIN_VEL_COEF;
        obs[idx++] = ballState.vel.z * LIN_VEL_COEF;
        obs[idx++] = ballState.angVel.x * ANG_VEL_COEF;
        obs[idx++] = ballState.angVel.y * ANG_VEL_COEF;
        obs[idx++] = ballState.angVel.z * ANG_VEL_COEF;

        // Boost Pad Timers (34 floats)
        for (int p = 0; p < boostPads.size(); ++p) {
            auto padState = boostPads[p]->GetState();
            obs[idx++] = padState.cooldown * PAD_TIMER_COEF;
        }

        // Agent-Specific Partial Obs (9 floats)
        obs[idx++] = static_cast<float>(carState.lastControls.jump);
        obs[idx++] = carState.handbrakeVal;
        obs[idx++] = static_cast<float>(carState.hasJumped);
        obs[idx++] = static_cast<float>(carState.isJumping);
        obs[idx++] = static_cast<float>(carState.hasFlipped);
        obs[idx++] = static_cast<float>(carState.isFlipping);
        obs[idx++] = static_cast<float>(carState.hasDoubleJumped);
        obs[idx++] = static_cast<float>(carState.HasFlipOrJump());
        obs[idx++] = carState.airTimeSinceJump;

        // idx is now 52

        // Self Car Obs (20 floats)
        std::array<float, CAR_OBS_SIZE> selfCarObs = _generateCarObs(car, inverted);
        std::copy(selfCarObs.begin(), selfCarObs.end(), obs.begin() + idx);
        idx += CAR_OBS_SIZE; // idx is now 72

        // Allies & Enemies
        std::vector<std::array<float, CAR_OBS_SIZE>> allies;
        std::vector<std::array<float, CAR_OBS_SIZE>> enemies;

        for (int j = 0; j < NUM_AGENTS; ++j) {
            if (i == j) continue; // Skip self

            RocketSim::Car* otherCar = cars[j];
            if (otherCar->team == car->team) {
                allies.push_back(_generateCarObs(otherCar, inverted));
            } else {
                enemies.push_back(_generateCarObs(otherCar, inverted));
            }
        }

        // Allies + Pad (1 * 20 = 20 floats)
        int alliesToCopy = 0;
        for (const auto& allyObs : allies) {
            if (alliesToCopy >= (ZERO_PADDING - 1)) break;
            std::copy(allyObs.begin(), allyObs.end(), obs.begin() + idx);
            idx += CAR_OBS_SIZE;
            alliesToCopy++;
        }
        idx += (ZERO_PADDING - 1 - alliesToCopy) * CAR_OBS_SIZE; // idx is now 92

        // Enemies + Pad (2 * 20 = 40 floats)
        int enemiesToCopy = 0;
        for (const auto& enemyObs : enemies) {
            if (enemiesToCopy >= ZERO_PADDING) break;
            std::copy(enemyObs.begin(), enemyObs.end(), obs.begin() + idx);
            idx += CAR_OBS_SIZE;
            enemiesToCopy++;
        }
        idx += (ZERO_PADDING - enemiesToCopy) * CAR_OBS_SIZE; // idx is now 132

    }

    return allObs;
}

void RLEnv::reset(std::array<std::array<float, 138>, 4> &obs) {
    _resetToKickoff();
    obs = _getObs();
}


void RLEnv::step(const std::array<int, 4>& actions, std::array<std::array<float, 138>, 4> &obs, float &reward, bool &terminated) {

    {
        TRACE_SCOPE("set_controls");
        for (int i = 0; i < NUM_AGENTS; ++i) {
            const std::array<float, 8>& controlsArray = m_lookupTable.at(actions[i]);
            RocketSim::CarControls controls = {};
            controls.throttle = controlsArray[0];
            controls.steer    = controlsArray[1];
            controls.pitch    = controlsArray[2];
            controls.yaw      = controlsArray[3];
            controls.roll     = controlsArray[4];
            controls.jump     = (bool)controlsArray[5];
            controls.boost    = (bool)controlsArray[6];
            controls.handbrake= (bool)controlsArray[7];
            cars[i]->controls = controls;
        }
    }

    {
        TRACE_SCOPE("arena_step");
        // tick skip
        arena->Step(ACTION_REPEATS);
    }

    {
        TRACE_SCOPE("check_termination");
        bool goalScored = arena->IsBallScored();

        // Check for timeout (30 seconds * 120 ticks/sec = 3600 ticks)
        const uint64_t MAX_EPISODE_TICKS = 30 * 120;
        bool timeOut = (arena->tickCount > MAX_EPISODE_TICKS);

        terminated = goalScored || timeOut;

        reward = 0.0f; // placeholder
    }

    {
        TRACE_SCOPE("get_obs");
        obs = _getObs();
    }
}
