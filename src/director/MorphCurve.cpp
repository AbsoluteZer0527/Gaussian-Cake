#include "MorphCurve.h"
#include <algorithm>
#include <cmath>

// One coordinate of the Bezier with ends 0 and 1: 3(1-s)^2 s a + 3(1-s) s^2 b + s^3
static float Bez(float a, float b, float s) {
    float r = 1.0f - s;
    return 3.0f * r * r * s * a + 3.0f * r * s * s * b + s * s * s;
}

float MorphCurve::Eval(float x) const {
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    float x1 = std::clamp(p1.x, 0.0f, 1.0f), x2 = std::clamp(p2.x, 0.0f, 1.0f);
    float lo = 0.0f, hi = 1.0f;
    for (int i = 0; i < kCurveSteps; ++i) {
        float mid = 0.5f * (lo + hi);
        if (Bez(x1, x2, mid) < x) lo = mid; else hi = mid;
    }
    return Bez(p1.y, p2.y, 0.5f * (lo + hi));
}
