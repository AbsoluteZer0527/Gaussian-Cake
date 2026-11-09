#pragma once
#include <vector>

// Follow-through: one damped spring per point, on the point's own time (how
// far it is from A to B). The morph moves each spring's target; the point
// follows a little late, overshoots and settles, so parts wobble when they
// arrive. Working on time rather than position keeps the wobble along each
// point's own path, so the GPU blend (start + u (end - start)) still works.
class ScalarSprings {
public:
    // Stiffness and damping for a jiggle amount in [0, 1]. 0 is not a spring
    // at all: Step then returns the targets unchanged.
    static void Constants(float jiggle, float& k, float& c);

    // Jump to these values, at rest.
    void Reset(const std::vector<float>& values);

    // Advance dt seconds toward the targets. Resets first if the count changed.
    const std::vector<float>& Step(const std::vector<float>& targets, float dt, float jiggle);

    // Every spring is within eps of its target and nearly still.
    bool Settled(float eps = 1e-3f) const;

    const std::vector<float>& Values() const { return x; }

private:
    std::vector<float> x, v, target;
};
