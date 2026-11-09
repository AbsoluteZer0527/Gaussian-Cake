#include "MorphCurve.h"
#include "imgui.h"
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

bool DrawCurveEditor(const char* label, MorphCurve& curve, float size) {
    ImGui::PushID(label);
    ImGui::TextUnformatted(label);

    // y from -0.8 to 1.8: the handles sit well outside the curve they make
    // (a handle at 1.8 overshoots by ~0.17), so the canvas has room for them
    const float yLo = -0.8f, yHi = 1.8f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 canvas(size, size);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    auto toScreen = [&](glm::vec2 p) {
        return ImVec2(origin.x + p.x * canvas.x,
                      origin.y + (1.0f - (p.y - yLo) / (yHi - yLo)) * canvas.y);
    };
    auto fromScreen = [&](ImVec2 s) {
        return glm::vec2((s.x - origin.x) / canvas.x,
                         yLo + (1.0f - (s.y - origin.y) / canvas.y) * (yHi - yLo));
    };

    dl->AddRectFilled(origin, ImVec2(origin.x + canvas.x, origin.y + canvas.y), IM_COL32(30, 30, 36, 255));
    // the 0 and 1 levels: inside them is a plain ease, outside is anticipation / overshoot
    ImU32 band = IM_COL32(90, 90, 110, 255);
    dl->AddLine(toScreen({ 0.0f, 0.0f }), toScreen({ 1.0f, 0.0f }), band);
    dl->AddLine(toScreen({ 0.0f, 1.0f }), toScreen({ 1.0f, 1.0f }), band);

    // the curve itself
    const int n = 48;
    ImVec2 prev = toScreen({ 0.0f, 0.0f });
    for (int i = 1; i <= n; ++i) {
        float x = (float)i / n;
        ImVec2 p = toScreen({ x, curve.Eval(x) });
        dl->AddLine(prev, p, IM_COL32(255, 120, 190, 255), 2.0f);
        prev = p;
    }

    // handles: stems from the ends, then a draggable dot each
    bool changed = false;
    glm::vec2* handles[2] = { &curve.p1, &curve.p2 };
    glm::vec2 anchors[2] = { { 0.0f, 0.0f }, { 1.0f, 1.0f } };
    for (int h = 0; h < 2; ++h) {
        ImVec2 s = toScreen(*handles[h]);
        dl->AddLine(toScreen(anchors[h]), s, IM_COL32(160, 160, 180, 255));
        ImGui::SetCursorScreenPos(ImVec2(s.x - 6.0f, s.y - 6.0f));
        ImGui::InvisibleButton(h == 0 ? "##h1" : "##h2", ImVec2(12.0f, 12.0f));
        bool active = ImGui::IsItemActive();
        if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            glm::vec2 p = fromScreen(ImGui::GetIO().MousePos);
            p.x = std::clamp(p.x, 0.0f, 1.0f);
            p.y = std::clamp(p.y, yLo, yHi);
            if (p != *handles[h]) { *handles[h] = p; changed = true; }
        }
        bool hot = active || ImGui::IsItemHovered();
        dl->AddCircleFilled(s, hot ? 6.0f : 4.5f, hot ? IM_COL32(255, 220, 120, 255) : IM_COL32(240, 240, 240, 255));
    }

    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + canvas.y + ImGui::GetStyle().ItemSpacing.y));
    ImGui::Dummy(ImVec2(canvas.x, 0.0f));
    ImGui::PopID();
    return changed;
}
