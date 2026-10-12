#pragma once
#include "core.h"
#include "OptimalTransport.h"
#include <vector>

enum class AnimationState {
    GAUSSIAN_A,
    BAKING,
    GAUSSIAN_B
};

class TransportAnimator {
public:
    AnimationState state;
    float transportDuration;   // seconds for a full bake, default 3.0
    float transportProgress;   // position on the A -> B timeline, 0.0 -> 1.0

    enum class EasingMode { LINEAR, EASE_IN_OUT, BOUNCE };
    EasingMode easingMode;

    TransportAnimator();

    // Gaussian OT path: apply OTMap to each start position to get destinations.
    void Initialize(const std::vector<glm::vec3>& startPositions, const OTMap& map);

    // Discrete mesh OT path: start and end positions already matched externally.
    void InitializeWithEndPoints(const std::vector<glm::vec3>& start,
                                 const std::vector<glm::vec3>& end);

    //write interpolated position into outposition.
    void Update(float deltaTime, std::vector<glm::vec3>& outPositions);

    //A -> B transport. Only fires from GAUSSIAN_A.
    void StartTransport();

    //B -> A return: plays the same timeline backwards. Only fires from GAUSSIAN_B.
    void ReverseTransport();

    //reset back to GAUSSIAN_A positions.
    void Reset();

    bool IsTransporting() const { return state == AnimationState::BAKING; }
    float GetProgress()   const { return transportProgress; }
    bool  IsReversing()   const { return direction < 0.0f; }

    // Eased blend weight toward B for the current point on the timeline.
    // 0 = fully A, 1 = fully B, in both playback directions.
    float GetBlend()      const { return ApplyEasing(transportProgress); }

    // Exposed so callers (e.g., a parallel target-topology mesh) can lerp
    // their own buffers in sync with this animator's timeline.
    float ApplyEasing(float t) const;

    void DrawGUI();

private:
    std::vector<glm::vec3> startPos;
    std::vector<glm::vec3> destPos;
    float direction;           // +1 playing A -> B, -1 playing B -> A
};