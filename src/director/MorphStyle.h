#pragma once
#include "MorphCurve.h"
#include <string>
#include <vector>

// Where the morph starts spreading from.
enum class OriginMode {
    None    = 0,   // everything moves together
    Bottom  = 1,   // heat rises from the lowest points (the cake pan)
    Top     = 2,   // from the highest points (melting from the top)
    Clicked = 3,   // from the point the user clicked (Bottom until one is set)
};

// How a morph moves: the knobs a preset sets. Everything here is a "feel"
// setting; none of it changes which point goes where.
struct MorphStyle {
    std::string name = "Custom";
    OriginMode  origin = OriginMode::None;
    float wave = 0.0f;            // 0 = all points leave together, up to 0.8 = a slow wave
    MorphCurve early;             // timing curve of the points the wave reaches first
    MorphCurve late;              // ...and of the last ones (the rest blend between the two)
    float sag = 0.0f;             // Topology: mid-flight droop, in shape units
    float jiggle = 0.0f;          // Topology: 0 = arrive exactly, 1 = wobble and settle
    float cohesion = 0.6f;        // Splats: 0 = every splat flies alone, 1 = chunks
    float midGrow = 0.3f;         // Splats: mid-flight size boost
    bool  splatsBest = false;     // looks best in Splats mode (hint only)
    std::string hint;             // one line for the tooltip
};

// Classic, Bake, Ripple, Melt, Pop, Dissolve (Classic first: the default).
const std::vector<MorphStyle>& BuiltinStyles();

// Index of the built-in style with this name (case-insensitive), or -1.
int FindStyle(const std::string& name);

// One point's place on the A -> B path at timeline progress t in [0, 1]:
// the wave holds it until its delay comes up (WaveTime), then its curve
// (early and late blended by the delay) eases it. 0 at t = 0, 1 at t = 1,
// whatever the delay; may leave [0, 1] in between (anticipation, overshoot).
float VertexTime(float t, float wave, float delay, const MorphCurve& early, const MorphCurve& late);
