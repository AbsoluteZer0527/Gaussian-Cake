#include "MorphSpring.h"
#include <algorithm>
#include <cmath>

void ScalarSprings::Constants(float jiggle, float& k, float& c) {
    jiggle = std::clamp(jiggle, 0.0f, 1.0f);
    k = 400.0f + (60.0f - 400.0f) * jiggle;     // softer with more jiggle
    float zeta = 0.95f + (0.25f - 0.95f) * jiggle;   // less damping with more jiggle
    c = 2.0f * zeta * std::sqrt(k);
}

void ScalarSprings::Reset(const std::vector<float>& values) {
    x = values;
    target = values;
    v.assign(values.size(), 0.0f);
}

const std::vector<float>& ScalarSprings::Step(const std::vector<float>& targets, float dt, float jiggle) {
    if (x.size() != targets.size()) Reset(targets);
    target = targets;
    if (jiggle <= 1e-3f) {   // no spring: follow exactly
        x = targets;
        std::fill(v.begin(), v.end(), 0.0f);
        return x;
    }
    float k, c;
    Constants(jiggle, k, c);
    // semi-implicit Euler, substepped so the stiffest setting stays stable at 20 fps
    const int sub = 4;
    float h = std::clamp(dt, 0.0f, 0.05f) / sub;
    for (size_t i = 0; i < x.size(); ++i) {
        float xi = x[i], vi = v[i], g = targets[i];
        for (int s = 0; s < sub; ++s) {
            vi += h * (k * (g - xi) - c * vi);
            xi += h * vi;
        }
        x[i] = xi;
        v[i] = vi;
    }
    return x;
}

bool ScalarSprings::Settled(float eps) const {
    for (size_t i = 0; i < x.size(); ++i)
        if (std::abs(x[i] - target[i]) > eps || std::abs(v[i]) > 10.0f * eps) return false;
    return true;
}
