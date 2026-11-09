#pragma once
#include <glm/glm.hpp>

// A timing curve for the morph: a cubic Bezier from (0, 0) to (1, 1) with two
// handles p1 and p2, the same shape as CSS cubic-bezier(). x is the time, y how
// far the shape has got. y may leave [0, 1] near the ends: below 0 is
// anticipation (a small pull back before moving), above 1 is overshoot.
//
// Eval solves x(s) = x by bisection (a fixed number of steps, so splat_prep.comp
// can repeat it line for line and get the same answer), then returns y(s).
// The handles' x is kept in [0, 1], which keeps x(s) increasing.
struct MorphCurve {
    glm::vec2 p1 = glm::vec2(0.0f, 0.0f);
    glm::vec2 p2 = glm::vec2(1.0f, 1.0f);

    MorphCurve() = default;
    MorphCurve(float x1, float y1, float x2, float y2) : p1(x1, y1), p2(x2, y2) {}

    // Exact at the ends: Eval(0) = 0, Eval(1) = 1. x outside [0, 1] is clamped.
    float Eval(float x) const;

    bool operator==(const MorphCurve& o) const { return p1 == o.p1 && p2 == o.p2; }
    bool operator!=(const MorphCurve& o) const { return !(*this == o); }

    static MorphCurve Linear() { return MorphCurve(0.0f, 0.0f, 1.0f, 1.0f); }
};

// Bisection steps in Eval (and in splat_prep.comp's CurveEval).
constexpr int kCurveSteps = 20;
