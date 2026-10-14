#include "TransportAnimator.h"
#include "imgui.h"
#include <cmath>
#include <algorithm>

TransportAnimator::TransportAnimator() {
    state = AnimationState::GAUSSIAN_A;
    transportDuration = 3.0f;
    transportProgress = 0.0f;
    direction = 1.0f;
    easingMode = EasingMode::EASE_IN_OUT;
}

void TransportAnimator::Initialize(const std::vector<glm::vec3>& startPositions, const OTMap& map) {
    startPos = startPositions;
    destPos.clear();
    destPos.reserve(startPositions.size());

    //compute OT of every p and add them to the map.
    for (const auto& p : startPositions){
        destPos.push_back(OptimalTransport::Apply(map, p));
    }

    state = AnimationState::GAUSSIAN_A;
    transportProgress = 0.0f;
    direction = 1.0f;
}

void TransportAnimator::Update(float deltaTime, std::vector<glm::vec3>& outPositions) {
    if (state != AnimationState::BAKING) return;

    transportProgress += direction * deltaTime / transportDuration;

    //reaching either end of the timeline settles into that state
    if (transportProgress >= 1.0f) {
        transportProgress = 1.0f;
        state = AnimationState::GAUSSIAN_B;
    } else if (transportProgress <= 0.0f) {
        transportProgress = 0.0f;
        state = AnimationState::GAUSSIAN_A;
    }

    float t = GetBlend();

    outPositions.resize(startPos.size());
    for (int i = 0; i < (int)startPos.size(); i++){
        outPositions[i] = glm::mix(startPos[i], destPos[i], t);
    }
}

void TransportAnimator::InitializeWithEndPoints(const std::vector<glm::vec3>& start,
                                                const std::vector<glm::vec3>& end) {
    startPos = start;
    destPos  = end;
    state = AnimationState::GAUSSIAN_A;
    transportProgress = 0.0f;
    direction = 1.0f;
}

void TransportAnimator::StartTransport() {
    if (state != AnimationState::GAUSSIAN_A) return;
    transportProgress = 0.0f;
    direction = 1.0f;
    state = AnimationState::BAKING;
}

void TransportAnimator::ReverseTransport() {
    if (state != AnimationState::GAUSSIAN_B) return;
    //play the same timeline backwards; endpoints stay A -> B
    transportProgress = 1.0f;
    direction = -1.0f;
    state = AnimationState::BAKING;
}

void TransportAnimator::Reset() {
    state = AnimationState::GAUSSIAN_A;
    transportProgress = 0.0f;
    direction = 1.0f;
}

float TransportAnimator::ApplyEasing(float t) const {
    //fun variation of the animations.
    switch (easingMode) {
        case EasingMode::LINEAR:
            return t;

        case EasingMode::EASE_IN_OUT:
            // Quadratic ease-in-out: slow start, fast middle, slow end
            return t < 0.5f
                ? 2.0f * t * t
                : 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;

        case EasingMode::BOUNCE: {
            // Ease in, then a small overshoot and settle
            if (t < 0.8f) {
                float s = t / 0.8f;
                return 0.9f * s * s;
            } else {
                float s = (t - 0.8f) / 0.2f;
                // Goes to 1.08 then back to 1.0
                return 0.9f + 0.18f * s - 0.08f * s * s;
            }
        }
    }
    return t;
}

void TransportAnimator::DrawParamsGUI() {
    ImGui::SliderFloat("Duration (s)", &transportDuration, 0.5f, 10.0f, "%.1f");

    const char* modes[] = {"Linear", "Ease In/Out", "Bounce"};
    int modeIdx = (int)easingMode;
    if (ImGui::Combo("Easing", &modeIdx, modes, 3))
        easingMode = (EasingMode)modeIdx;
}

void TransportAnimator::DrawStatusGUI() {
    //one bar for the whole A -> B timeline, filled to the current position
    const char* label =
        state == AnimationState::GAUSSIAN_A ? "At A" :
        state == AnimationState::GAUSSIAN_B ? "At B" :
        IsReversing() ? "Reverting to A..." : "Morphing to B...";
    ImGui::ProgressBar(transportProgress, ImVec2(-1, 0), label);
}
