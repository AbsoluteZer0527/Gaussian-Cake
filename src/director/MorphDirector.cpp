#include "MorphDirector.h"
#include "SplatHeat.h"
#include <algorithm>
#include <cmath>

// Typical spacing between neighbouring points, from a sample of them.
static float PointSpacing(const std::vector<glm::vec3>& pts) {
    glm::vec3 lo(1e9f), hi(-1e9f);
    for (const glm::vec3& p : pts) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
    float diag = glm::length(hi - lo);
    if (pts.size() < 2 || diag <= 0.0f) return 0.05f;
    float guess = 2.0f * diag / std::sqrt((float)pts.size());   // points on a surface
    PointGrid g;
    g.Build(pts, guess);
    const int samples = std::min<int>(500, (int)pts.size());
    const int stride = std::max<int>(1, (int)pts.size() / samples);
    double sum = 0.0;
    int found = 0;
    for (size_t i = 0; i < pts.size(); i += stride) {
        float best = 1e30f;
        g.ForEachNear(pts[i], [&](int j) {
            if (j != (int)i) best = std::min(best, Dist2(pts[i], pts[j]));
        });
        if (best < 1e29f) { sum += std::sqrt(best); ++found; }
    }
    return found ? (float)(sum / found) : guess;
}

void MorphDirector::Bind(const std::vector<glm::vec3>& pts, float cellSize, int newSlot) {
    // keep this slot's paint with the points it was painted on
    auto findSlot = [&](int k) {
        for (auto& e : savedPaint) if (e.first == k) return &e.second;
        return (SavedPaint*)nullptr;
    };
    if (!points.empty()) {
        SavedPaint* sp = findSlot(slot);
        if (!sp) { savedPaint.push_back({ slot, SavedPaint() }); sp = &savedPaint.back().second; }
        if (hasPaint) { sp->points = points; sp->paint = paint; }
        else          { sp->points.clear(); sp->paint.clear(); }
    }

    points = pts;
    slot = newSlot;
    size_t n = points.size();
    heat.assign(n, 0.0f);
    paint.assign(n, 0.0f);
    delay.assign(n, 0.0f);
    hasPaint = false;
    if (SavedPaint* sp = findSlot(slot)) {
        if (!sp->points.empty() && n > 0) {
            PointGrid old;
            old.Build(sp->points, 3.0f * PointSpacing(sp->points));
            for (size_t i = 0; i < n; ++i) {
                int j = old.Nearest(points[i]);
                if (j >= 0) paint[i] = sp->paint[j];
                if (paint[i] != 0.0f) hasPaint = true;
            }
        }
    }
    solvedMode = -1;
    springs = ScalarSprings();
    if (n == 0) return;
    // the heat solve's k-nearest-neighbour search looks one cell around each
    // point, so a few spacings per cell finds 8 neighbours
    cell = cellSize > 0.0f ? cellSize : 3.0f * PointSpacing(points);
    grid.Build(points, cell);
    ++delayVersion;
}

void MorphDirector::UpdateDelay(OriginMode mode) {
    if (points.empty()) return;
    if (mode == OriginMode::Clicked && !hasOrigin) mode = OriginMode::Bottom;
    if ((int)mode == solvedMode && (mode != OriginMode::Clicked || origin == solvedOrigin)) return;

    if (mode == OriginMode::None) {
        std::fill(heat.begin(), heat.end(), 0.0f);
    } else {
        size_t n = points.size();
        glm::vec3 o = mode == OriginMode::Clicked ? origin : glm::vec3(0.0f);
        bool cached = false;
        for (const CachedHeat& c : heatCache) {
            if (c.n == n && c.first == points[0] && c.middle == points[n / 2] &&
                c.mode == (int)mode && c.origin == o) {
                heat = c.heat;
                cached = true;
                break;
            }
        }
        if (!cached) {
            WaveSeed seed = mode == OriginMode::Top ? WaveSeed::Top
                          : mode == OriginMode::Clicked ? WaveSeed::Point : WaveSeed::Bottom;
            heat = HeatWaveDelay(points, cell, sigma, seedFraction, seed, origin);
            if (heatCache.size() >= 8) heatCache.erase(heatCache.begin());
            heatCache.push_back({ n, points[0], points[n / 2], (int)mode, o, heat });
        }
    }
    solvedMode = (int)mode;
    solvedOrigin = origin;
    Combine();
}

void MorphDirector::Combine() {
    for (size_t i = 0; i < points.size(); ++i)
        delay[i] = std::clamp(heat[i] + paint[i], 0.0f, 1.0f);
    ++delayVersion;
}

void MorphDirector::Paint(const glm::vec3& center, float sign) {
    if (points.empty()) return;
    float r = brushRadius, r2 = r * r;
    for (size_t i = 0; i < points.size(); ++i) {
        float d2 = Dist2(points[i], center);
        if (d2 >= r2) continue;
        float f = 1.0f - std::sqrt(d2) / r;   // soft edge
        paint[i] = std::clamp(paint[i] + sign * brushStrength * f * f, -1.0f, 1.0f);
        hasPaint = true;
    }
    Combine();
}

void MorphDirector::ClearPaint() {
    std::fill(paint.begin(), paint.end(), 0.0f);
    hasPaint = false;
    Combine();
}

const std::vector<float>& MorphDirector::Evaluate(float local, float dt, const MorphStyle& style) {
    size_t n = points.size();
    targets.resize(n);
    for (size_t i = 0; i < n; ++i)
        targets[i] = VertexTime(local, style.wave, delay[i], style.early, style.late);
    if (dt <= 0.0f) {
        springs.Reset(targets);
        return springs.Values();
    }
    return springs.Step(targets, dt, style.jiggle);
}

void MorphDirector::Snap(float local, const MorphStyle& style) {
    Evaluate(local, 0.0f, style);
}

std::vector<glm::vec3> MorphDirector::OverlayColors(const std::vector<glm::vec3>& displayed,
                                                    const std::vector<glm::vec3>& pinsOn) const {
    // early = warm orange, late = cool blue, as display colours
    const glm::vec3 early(1.0f, 0.45f, 0.15f), late(0.2f, 0.45f, 1.0f);
    std::vector<glm::vec3> out(displayed.size(), glm::vec3(0.8f));
    bool haveDelay = !points.empty() && delay.size() == points.size();
    float dotR2 = 0.0016f;   // 0.04 units
    for (size_t i = 0; i < displayed.size(); ++i) {
        const glm::vec3& p = displayed[i];
        if (haveDelay) {
            int j = grid.Nearest(p);
            if (j >= 0) out[i] = glm::mix(early, late, delay[j]);
        }
        if (hasOrigin && Dist2(p, origin) < dotR2 * 4.0f) out[i] = glm::vec3(1.0f, 1.0f, 0.3f);
        for (const glm::vec3& pin : pinsOn)
            if (Dist2(p, pin) < dotR2) out[i] = glm::vec3(0.2f, 1.0f, 0.4f);
        if (pinWaitingForB && Dist2(p, pendingA) < dotR2) out[i] = glm::vec3(0.2f, 1.0f, 0.4f);
    }
    return out;
}
