#include "DirectorWidgets.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

static const float kPi = 3.14159265f;

ImU32 AccentColor() { return IM_COL32(255, 102, 179, 255); }

static ImU32 Col(const glm::vec3& c, float a) {
    return ImGui::ColorConvertFloat4ToU32(ImVec4(c.x, c.y, c.z, a));
}

ImU32 StyleColor(const MorphStyle& style, float alpha) { return Col(style.color, alpha); }

// ---- icons ----

static void Arrow(ImDrawList* dl, ImVec2 c, float s, float dirY, ImU32 col) {
    // a stem and a head, pointing up (dirY = -1) or down (+1)
    dl->AddLine(ImVec2(c.x, c.y - dirY * 0.4f * s), ImVec2(c.x, c.y + dirY * 0.1f * s), col, 2.0f);
    dl->AddTriangleFilled(ImVec2(c.x, c.y + dirY * 0.45f * s), ImVec2(c.x - 0.3f * s, c.y + dirY * 0.05f * s),
                          ImVec2(c.x + 0.3f * s, c.y + dirY * 0.05f * s), col);
}

void DrawIcon(ImDrawList* dl, Icon icon, ImVec2 c, float s, ImU32 col) {
    float h = 0.5f * s;
    switch (icon) {
        case Icon::Play:
            dl->AddTriangleFilled(ImVec2(c.x - 0.3f * s, c.y - 0.4f * s), ImVec2(c.x + 0.4f * s, c.y),
                                  ImVec2(c.x - 0.3f * s, c.y + 0.4f * s), col);
            break;
        case Icon::Pause:
            dl->AddRectFilled(ImVec2(c.x - 0.35f * s, c.y - 0.4f * s), ImVec2(c.x - 0.08f * s, c.y + 0.4f * s), col, 1.0f);
            dl->AddRectFilled(ImVec2(c.x + 0.08f * s, c.y - 0.4f * s), ImVec2(c.x + 0.35f * s, c.y + 0.4f * s), col, 1.0f);
            break;
        case Icon::ToStart:
            dl->AddRectFilled(ImVec2(c.x - 0.4f * s, c.y - 0.35f * s), ImVec2(c.x - 0.25f * s, c.y + 0.35f * s), col);
            dl->AddTriangleFilled(ImVec2(c.x + 0.35f * s, c.y - 0.35f * s), ImVec2(c.x - 0.2f * s, c.y),
                                  ImVec2(c.x + 0.35f * s, c.y + 0.35f * s), col);
            break;
        case Icon::ToEnd:
            dl->AddRectFilled(ImVec2(c.x + 0.25f * s, c.y - 0.35f * s), ImVec2(c.x + 0.4f * s, c.y + 0.35f * s), col);
            dl->AddTriangleFilled(ImVec2(c.x - 0.35f * s, c.y - 0.35f * s), ImVec2(c.x + 0.2f * s, c.y),
                                  ImVec2(c.x - 0.35f * s, c.y + 0.35f * s), col);
            break;
        case Icon::Loop: {
            float r = 0.32f * s;
            dl->PathArcTo(c, r, 0.15f * kPi, 1.75f * kPi, 16);
            dl->PathStroke(col, 0, 2.0f);
            ImVec2 e(c.x + r * std::cos(0.15f * kPi), c.y + r * std::sin(0.15f * kPi));
            dl->AddTriangleFilled(ImVec2(e.x - 0.2f * s, e.y), ImVec2(e.x + 0.2f * s, e.y),
                                  ImVec2(e.x, e.y + 0.22f * s), col);
            break;
        }
        case Icon::Navigate: {
            // move cross: four arrow heads
            dl->AddLine(ImVec2(c.x - 0.38f * s, c.y), ImVec2(c.x + 0.38f * s, c.y), col, 1.5f);
            dl->AddLine(ImVec2(c.x, c.y - 0.38f * s), ImVec2(c.x, c.y + 0.38f * s), col, 1.5f);
            float a = 0.45f * s, b = 0.25f * s, w = 0.13f * s;
            dl->AddTriangleFilled(ImVec2(c.x, c.y - a), ImVec2(c.x - w, c.y - b), ImVec2(c.x + w, c.y - b), col);
            dl->AddTriangleFilled(ImVec2(c.x, c.y + a), ImVec2(c.x + w, c.y + b), ImVec2(c.x - w, c.y + b), col);
            dl->AddTriangleFilled(ImVec2(c.x - a, c.y), ImVec2(c.x - b, c.y + w), ImVec2(c.x - b, c.y - w), col);
            dl->AddTriangleFilled(ImVec2(c.x + a, c.y), ImVec2(c.x + b, c.y - w), ImVec2(c.x + b, c.y + w), col);
            break;
        }
        case Icon::StartPoint:
            dl->AddCircle(c, 0.3f * s, col, 16, 1.5f);
            dl->AddCircleFilled(c, 0.1f * s, col);
            dl->AddLine(ImVec2(c.x, c.y - h), ImVec2(c.x, c.y - 0.3f * s), col, 1.5f);
            dl->AddLine(ImVec2(c.x, c.y + h), ImVec2(c.x, c.y + 0.3f * s), col, 1.5f);
            dl->AddLine(ImVec2(c.x - h, c.y), ImVec2(c.x - 0.3f * s, c.y), col, 1.5f);
            dl->AddLine(ImVec2(c.x + h, c.y), ImVec2(c.x + 0.3f * s, c.y), col, 1.5f);
            break;
        case Icon::Brush:
            // handle from the top right, bristles at the bottom left
            dl->AddLine(ImVec2(c.x + 0.42f * s, c.y - 0.42f * s), ImVec2(c.x - 0.02f * s, c.y + 0.02f * s), col, 2.5f);
            dl->AddCircleFilled(ImVec2(c.x - 0.18f * s, c.y + 0.18f * s), 0.17f * s, col);
            dl->AddTriangleFilled(ImVec2(c.x - 0.33f * s, c.y + 0.22f * s), ImVec2(c.x - 0.22f * s, c.y + 0.33f * s),
                                  ImVec2(c.x - 0.45f * s, c.y + 0.45f * s), col);
            break;
        case Icon::Pin:
            dl->AddCircleFilled(ImVec2(c.x, c.y - 0.2f * s), 0.2f * s, col);
            dl->AddRectFilled(ImVec2(c.x - 0.25f * s, c.y - 0.02f * s), ImVec2(c.x + 0.25f * s, c.y + 0.05f * s), col, 1.0f);
            dl->AddLine(ImVec2(c.x, c.y + 0.05f * s), ImVec2(c.x, c.y + 0.45f * s), col, 1.5f);
            break;
        case Icon::Plus:
            dl->AddLine(ImVec2(c.x - 0.32f * s, c.y), ImVec2(c.x + 0.32f * s, c.y), col, 2.0f);
            dl->AddLine(ImVec2(c.x, c.y - 0.32f * s), ImVec2(c.x, c.y + 0.32f * s), col, 2.0f);
            break;
        case Icon::Clear:
            dl->AddLine(ImVec2(c.x - 0.25f * s, c.y - 0.25f * s), ImVec2(c.x + 0.25f * s, c.y + 0.25f * s), col, 2.0f);
            dl->AddLine(ImVec2(c.x + 0.25f * s, c.y - 0.25f * s), ImVec2(c.x - 0.25f * s, c.y + 0.25f * s), col, 2.0f);
            break;
        case Icon::Link:   // two chain links, tilted
            dl->AddCircle(ImVec2(c.x - 0.17f * s, c.y + 0.12f * s), 0.22f * s, col, 16, 2.0f);
            dl->AddCircle(ImVec2(c.x + 0.17f * s, c.y - 0.12f * s), 0.22f * s, col, 16, 2.0f);
            break;
        case Icon::SpreadAll:
            dl->AddCircleFilled(c, 0.32f * s, col);
            break;
        case Icon::SpreadUp:
            Arrow(dl, c, s, -1.0f, col);
            break;
        case Icon::SpreadDown:
            Arrow(dl, c, s, 1.0f, col);
            break;
        case Icon::SpreadPoint:
            dl->AddCircleFilled(c, 0.1f * s, col);
            dl->AddCircle(c, 0.25f * s, col, 16, 1.2f);
            dl->AddCircle(c, 0.42f * s, col, 20, 1.2f);
            break;
        case Icon::Sooner:   // a fast-forward: earlier
            dl->AddTriangleFilled(ImVec2(c.x - 0.05f * s, c.y - 0.3f * s), ImVec2(c.x - 0.45f * s, c.y),
                                  ImVec2(c.x - 0.05f * s, c.y + 0.3f * s), col);
            dl->AddTriangleFilled(ImVec2(c.x + 0.35f * s, c.y - 0.3f * s), ImVec2(c.x - 0.05f * s, c.y),
                                  ImVec2(c.x + 0.35f * s, c.y + 0.3f * s), col);
            break;
        case Icon::Later:
            dl->AddTriangleFilled(ImVec2(c.x + 0.05f * s, c.y - 0.3f * s), ImVec2(c.x + 0.45f * s, c.y),
                                  ImVec2(c.x + 0.05f * s, c.y + 0.3f * s), col);
            dl->AddTriangleFilled(ImVec2(c.x - 0.35f * s, c.y - 0.3f * s), ImVec2(c.x + 0.05f * s, c.y),
                                  ImVec2(c.x - 0.35f * s, c.y + 0.3f * s), col);
            break;
    }
}

bool IconButton(const char* id, Icon icon, bool active, const char* tooltip, float size) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool pressed = ImGui::InvisibleButton(id, ImVec2(size, size));
    bool hovered = ImGui::IsItemHovered(), held = ImGui::IsItemActive();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImU32 bg = active ? AccentColor()
             : held   ? ImGui::GetColorU32(ImGuiCol_ButtonActive)
             : hovered ? ImGui::GetColorU32(ImGuiCol_ButtonHovered)
                       : ImGui::GetColorU32(ImGuiCol_FrameBg);
    dl->AddRectFilled(p, ImVec2(p.x + size, p.y + size), bg, 4.0f);
    ImU32 fg = ImGui::GetColorU32(ImGuiCol_Text);   // faded by BeginDisabled
    DrawIcon(dl, icon, ImVec2(p.x + 0.5f * size, p.y + 0.5f * size), 0.62f * size, fg);
    if (tooltip && *tooltip) ImGui::SetItemTooltip("%s", tooltip);
    return pressed;
}

bool IconSegmented(const char* id, int& value, const std::vector<SegmentOption>& options, float size) {
    bool changed = false;
    ImGui::PushID(id);
    for (int i = 0; i < (int)options.size(); ++i) {
        if (i > 0) ImGui::SameLine(0.0f, 2.0f);
        ImGui::PushID(i);
        if (IconButton("##seg", options[i].icon, value == i, options[i].tooltip, size) && value != i) {
            value = i;
            changed = true;
        }
        ImGui::PopID();
    }
    ImGui::PopID();
    return changed;
}

void Badge(const char* text, ImU32 bg, const char* tooltip) {
    ImVec2 ts = ImGui::CalcTextSize(text);
    ImVec2 pad(6.0f, 2.0f);
    ImVec2 p = ImGui::GetCursorScreenPos();
    float hFrame = ImGui::GetFrameHeight();
    p.y += 0.5f * (hFrame - ts.y - 2.0f * pad.y);
    ImGui::Dummy(ImVec2(ts.x + 2.0f * pad.x, hFrame));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + ts.x + 2.0f * pad.x, p.y + ts.y + 2.0f * pad.y), bg, 8.0f);
    dl->AddText(ImVec2(p.x + pad.x, p.y + pad.y), IM_COL32(20, 20, 24, 255), text);
    if (tooltip && *tooltip) ImGui::SetItemTooltip("%s", tooltip);
}

// ---- the timeline track ----

static std::string MeshName(const std::vector<std::string>& names, int mesh) {
    if (mesh <= 0 || mesh > (int)names.size()) return "Gaussian blob";
    return names[mesh - 1];
}

void TimelineTrack(const char* id, const MorphTimeline& tl, float progress, bool editKeys,
                   const std::vector<std::string>& meshNames, TrackResult& out) {
    ImGui::PushID(id);
    const float plusW = editKeys ? ImGui::GetFrameHeight() : 0.0f;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float rulerH = 16.0f, clipH = 26.0f, padX = 8.0f;
    float width = std::max(80.0f, ImGui::GetContentRegionAvail().x - (editKeys ? plusW + spacing : 0.0f));
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 size(width, rulerH + clipH + 6.0f);

    ImGui::InvisibleButton("##track", size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    float total = std::max(tl.Total(), 0.01f);
    float x0 = p0.x + padX, x1 = p0.x + width - padX;
    auto timeToX = [&](float t) { return x0 + (x1 - x0) * (t / total); };
    auto xToTime = [&](float x) { return std::clamp((x - x0) / (x1 - x0), 0.0f, 1.0f) * total; };
    float trackY0 = p0.y + rulerH + 2.0f, trackY1 = trackY0 + clipH;
    float midY = 0.5f * (trackY0 + trackY1);

    // background and ruler
    dl->AddRectFilled(p0, ImVec2(p0.x + size.x, p0.y + size.y), ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f);
    float step = total > 12.0f ? 2.0f : total > 6.0f ? 1.0f : 0.5f;
    for (float t = 0.0f; t <= total + 1e-3f; t += step) {
        float x = timeToX(t);
        bool whole = std::fabs(t - std::round(t)) < 1e-3f;
        dl->AddLine(ImVec2(x, p0.y + (whole ? 3.0f : 9.0f)), ImVec2(x, p0.y + rulerH), IM_COL32(120, 120, 135, 255));
        if (whole) {
            char lbl[16];
            snprintf(lbl, sizeof(lbl), "%gs", t);
            dl->AddText(ImVec2(x + 3.0f, p0.y + 1.0f), IM_COL32(150, 150, 165, 255), lbl);
        }
    }

    // what's under the mouse
    ImVec2 m = ImGui::GetIO().MousePos;
    int hoverKey = -1, hoverSeg = -1;
    if (hovered || active) {
        for (int k = 0; k < (int)tl.keys.size(); ++k)
            if (std::fabs(m.x - timeToX(tl.KeyTime(k))) < 7.0f && std::fabs(m.y - midY) < 10.0f) hoverKey = k;
        if (hoverKey < 0 && m.y >= trackY0 && m.y <= trackY1)
            for (int s = 0; s < (int)tl.segs.size(); ++s)
                if (m.x >= timeToX(tl.KeyTime(s)) && m.x <= timeToX(tl.KeyTime(s + 1))) hoverSeg = s;
    }

    // clips
    for (int s = 0; s < (int)tl.segs.size(); ++s) {
        const MorphSegmentSpec& seg = tl.segs[s];
        float a = timeToX(tl.KeyTime(s)) + 6.0f, b = timeToX(tl.KeyTime(s + 1)) - 6.0f;
        if (b <= a) continue;
        bool sel = s == tl.selected;
        bool hot = s == hoverSeg;
        dl->AddRectFilled(ImVec2(a, trackY0), ImVec2(b, trackY1), StyleColor(seg.style, sel ? 0.95f : hot ? 0.7f : 0.5f), 5.0f);
        if (sel) dl->AddRect(ImVec2(a, trackY0), ImVec2(b, trackY1), IM_COL32(255, 255, 255, 230), 5.0f, 0, 1.5f);
        // the clip's timing curve, faint, and its name
        ImU32 ink = sel ? IM_COL32(20, 20, 24, 255) : IM_COL32(235, 235, 240, 255);
        dl->PushClipRect(ImVec2(a + 3.0f, trackY0), ImVec2(b - 3.0f, trackY1), true);
        ImVec2 prev(a, trackY1 - 3.0f);
        for (int i = 1; i <= 24; ++i) {
            float x = i / 24.0f;
            float y = std::clamp(seg.style.early.Eval(x), -0.2f, 1.2f);
            ImVec2 q(a + (b - a) * x, trackY1 - 3.0f - (clipH - 6.0f) * y);
            dl->AddLine(prev, q, sel ? IM_COL32(20, 20, 24, 70) : IM_COL32(255, 255, 255, 50), 1.5f);
            prev = q;
        }
        std::string label = seg.style.name;
        if (!seg.pins.empty()) label += "  *" + std::to_string(seg.pins.size());
        dl->AddText(ImVec2(a + 6.0f, midY - 0.5f * ImGui::GetTextLineHeight()), ink, label.c_str());
        dl->PopClipRect();
    }

    // key diamonds
    for (int k = 0; k < (int)tl.keys.size(); ++k) {
        float x = timeToX(tl.KeyTime(k));
        float r = (k == hoverKey) ? 7.5f : 6.0f;
        ImVec2 pts[4] = { { x, midY - r }, { x + r, midY }, { x, midY + r }, { x - r, midY } };
        bool sel = k == tl.selectedKey;
        dl->AddConvexPolyFilled(pts, 4, sel ? AccentColor() : IM_COL32(235, 235, 240, 255));
        dl->AddPolyline(pts, 4, IM_COL32(20, 20, 24, 255), ImDrawFlags_Closed, 1.5f);
    }

    // playhead
    float px = timeToX(std::clamp(progress, 0.0f, 1.0f) * total);
    dl->AddLine(ImVec2(px, p0.y), ImVec2(px, p0.y + size.y), AccentColor(), 2.0f);
    dl->AddTriangleFilled(ImVec2(px - 5.0f, p0.y), ImVec2(px + 5.0f, p0.y), ImVec2(px, p0.y + 7.0f), AccentColor());

    // ---- interaction ----
    static int dragKey = -1;
    static bool scrubbing = false;
    if (ImGui::IsItemActivated() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        dragKey = -1;
        scrubbing = false;
        if (hoverKey >= 0) {
            out.selectKey = hoverKey;
            if (editKeys && hoverKey > 0 && hoverKey < (int)tl.keys.size() - 1) dragKey = hoverKey;
        } else if (hoverSeg >= 0) {
            out.selectSeg = hoverSeg;
        } else {
            scrubbing = true;   // the ruler or the empty track
        }
    }
    if (active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (dragKey >= 0 && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
            out.retimeKey = dragKey;
            out.retimeTime = xToTime(m.x);
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        } else if (scrubbing) {
            out.seek = true;
            out.progress = xToTime(m.x) / total;
        }
    }
    if (!active) { dragKey = -1; scrubbing = false; }

    // tooltips
    if (hovered && !active) {
        if (hoverKey >= 0) {
            ImGui::SetTooltip("%s  %s\n%s", hoverKey == 0 ? "Start" : ("Key " + std::to_string(hoverKey)).c_str(),
                              MeshName(meshNames, tl.keys[hoverKey]).c_str(),
                              editKeys && hoverKey > 0 && hoverKey < (int)tl.keys.size() - 1
                                  ? "Drag to retime, right-click for options" : "Right-click for options");
        } else if (hoverSeg >= 0) {
            const MorphSegmentSpec& s = tl.segs[hoverSeg];
            ImGui::SetTooltip("%s -> %s\n%s, %.1f s%s\nClick to edit, right-click for styles",
                              MeshName(meshNames, tl.keys[hoverSeg]).c_str(),
                              MeshName(meshNames, tl.keys[hoverSeg + 1]).c_str(), s.style.name.c_str(), s.duration,
                              s.pins.empty() ? "" : (", " + std::to_string(s.pins.size()) + " pins").c_str());
        } else {
            ImGui::SetTooltip("Click or drag to scrub");
        }
    }

    // context menus
    static int ctxKey = -1, ctxSeg = -1;
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ctxKey = hoverKey;
        ctxSeg = hoverSeg;
        if (ctxKey >= 0 || ctxSeg >= 0) ImGui::OpenPopup("##trackctx");
    }
    if (ImGui::BeginPopup("##trackctx")) {
        if (ctxKey >= 0 && ctxKey < (int)tl.keys.size()) {
            ImGui::TextDisabled("%s", MeshName(meshNames, tl.keys[ctxKey]).c_str());
            if (editKeys && ImGui::BeginMenu("Change shape")) {
                for (int i = 1; i <= (int)meshNames.size(); ++i)
                    if (ImGui::MenuItem(meshNames[i - 1].c_str(), nullptr, tl.keys[ctxKey] == i)) {
                        out.shapeKey = ctxKey;
                        out.shapeMesh = i;
                    }
                ImGui::EndMenu();
            }
            if (editKeys && ImGui::MenuItem("Delete key", nullptr, false, tl.keys.size() > 2)) out.deleteKey = ctxKey;
        } else if (ctxSeg >= 0 && ctxSeg < (int)tl.segs.size()) {
            ImGui::TextDisabled("Style");
            const auto& styles = BuiltinStyles();
            for (int i = 0; i < (int)styles.size(); ++i)
                if (ImGui::MenuItem(styles[i].name.c_str(), nullptr, tl.segs[ctxSeg].preset == i)) {
                    out.styleSeg = ctxSeg;
                    out.styleIndex = i;
                }
        }
        ImGui::EndPopup();
    }

    // add a key at the end
    if (editKeys) {
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorScreenPos().x, p0.y + rulerH + 2.0f + 0.5f * (clipH - plusW)));
        if (IconButton("##addkey", Icon::Plus, false, "Add a shape at the end", plusW)) out.addKey = true;
        ImGui::SetCursorScreenPos(ImVec2(p0.x, p0.y + size.y + ImGui::GetStyle().ItemSpacing.y));
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
    }
    ImGui::PopID();
}

// ---- style tiles ----

static void PlotCurve(ImDrawList* dl, const MorphCurve& c, ImVec2 lo, ImVec2 hi, float yLo, float yHi,
                      float shift, ImU32 col, float thick) {
    ImVec2 prev;
    for (int i = 0; i <= 32; ++i) {
        float x = i / 32.0f;
        float y = c.Eval(std::clamp(x - shift, 0.0f, 1.0f) / std::max(1.0f - shift, 0.01f));
        ImVec2 q(lo.x + (hi.x - lo.x) * x, hi.y - (hi.y - lo.y) * (y - yLo) / (yHi - yLo));
        if (i > 0) dl->AddLine(prev, q, col, thick);
        prev = q;
    }
}

bool StyleTile(const char* id, const MorphStyle& style, bool selected, float size) {
    float textH = ImGui::GetTextLineHeight();
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool pressed = ImGui::InvisibleButton(id, ImVec2(size, size + textH + 2.0f));
    bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 q(p.x + size, p.y + size);
    dl->AddRectFilled(p, q, hovered ? ImGui::GetColorU32(ImGuiCol_FrameBgHovered) : ImGui::GetColorU32(ImGuiCol_FrameBg), 6.0f);
    if (selected) dl->AddRect(p, q, AccentColor(), 6.0f, 0, 2.0f);
    dl->AddRectFilled(ImVec2(p.x + 4.0f, q.y - 4.0f), ImVec2(q.x - 4.0f, q.y - 2.0f), StyleColor(style), 1.0f);

    // the timing: the trailing parts' curve shifted later by the spread
    ImVec2 lo(p.x + 6.0f, p.y + 6.0f), hi(q.x - 6.0f, q.y - 9.0f);
    if (style.wave > 0.0f) {
        PlotCurve(dl, style.early, lo, hi, -0.3f, 1.3f, 0.0f, StyleColor(style, 0.9f), 1.8f);
        PlotCurve(dl, style.late, lo, hi, -0.3f, 1.3f, 0.5f * style.wave, StyleColor(style, 0.45f), 1.5f);
    } else {
        PlotCurve(dl, style.early, lo, hi, -0.3f, 1.3f, 0.0f, StyleColor(style, 0.9f), 1.8f);
    }
    // name, centred under the tile
    ImVec2 ts = ImGui::CalcTextSize(style.name.c_str());
    dl->PushClipRect(ImVec2(p.x - 4.0f, q.y), ImVec2(q.x + 4.0f, q.y + textH + 4.0f), true);
    dl->AddText(ImVec2(p.x + 0.5f * (size - ts.x), q.y + 2.0f),
                selected ? ImGui::GetColorU32(ImGuiCol_Text) : ImGui::GetColorU32(ImGuiCol_TextDisabled),
                style.name.c_str());
    dl->PopClipRect();
    if (!style.hint.empty()) ImGui::SetItemTooltip("%s\n%s", style.name.c_str(), style.hint.c_str());
    return pressed;
}

// ---- easing editor ----

bool EasingEditor(const char* id, MorphCurve& lead, MorphCurve& trail, bool& linked, int& editing) {
    ImGui::PushID(id);
    bool changed = false;
    const ImU32 colLead = AccentColor(), colTrail = IM_COL32(90, 160, 255, 255);

    // legend chips: which curve the handles move, and the link
    auto chip = [&](const char* label, ImU32 col, bool on, const char* tip) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float h = ImGui::GetFrameHeight();
        float w = ImGui::CalcTextSize(label).x + h + 4.0f;
        bool pressed = ImGui::InvisibleButton(label, ImVec2(w, h));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        if (on || ImGui::IsItemHovered())
            dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), ImGui::GetColorU32(on ? ImGuiCol_FrameBgActive : ImGuiCol_FrameBgHovered), 4.0f);
        dl->AddCircleFilled(ImVec2(p.x + 0.5f * h, p.y + 0.5f * h), 0.22f * h, col);
        dl->AddText(ImVec2(p.x + h, p.y + 0.5f * (h - ImGui::GetTextLineHeight())),
                    ImGui::GetColorU32(on ? ImGuiCol_Text : ImGuiCol_TextDisabled), label);
        ImGui::SetItemTooltip("%s", tip);
        return pressed;
    };
    if (chip("Lead", colLead, linked || editing == 0, "The parts the change reaches first")) editing = 0;
    ImGui::SameLine(0.0f, 2.0f);
    if (chip("Trail", colTrail, linked || editing == 1, "The parts it reaches last")) editing = 1;
    ImGui::SameLine();
    if (IconButton("##link", Icon::Link, linked, linked ? "Linked: one curve for all parts (click to unlink)"
                                                         : "Link: use one curve for all parts", ImGui::GetFrameHeight())) {
        linked = !linked;
        if (linked) { trail = lead; changed = true; }
    }

    // the canvas: tall enough for both handles (the range is held while a
    // handle is dragged, so the curve doesn't slide under the mouse)
    static bool dragging = false;
    static float heldLo = -0.3f, heldHi = 1.3f;
    float yLo = -0.3f, yHi = 1.3f;
    for (const MorphCurve* c : { &lead, &trail }) {
        yLo = std::min({ yLo, c->p1.y - 0.15f, c->p2.y - 0.15f });
        yHi = std::max({ yHi, c->p1.y + 0.15f, c->p2.y + 0.15f });
    }
    if (dragging) { yLo = heldLo; yHi = heldHi; }
    heldLo = yLo;
    heldHi = yHi;
    bool anyActive = false;
    float w = ImGui::GetContentRegionAvail().x, h = std::min(130.0f, 0.6f * w);
    ImVec2 o = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##canvas", ImVec2(w, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float pad = 10.0f;
    auto toScreen = [&](glm::vec2 v) {
        return ImVec2(o.x + pad + v.x * (w - 2.0f * pad), o.y + pad + (1.0f - (v.y - yLo) / (yHi - yLo)) * (h - 2.0f * pad));
    };
    auto fromScreen = [&](ImVec2 s) {
        return glm::vec2((s.x - o.x - pad) / (w - 2.0f * pad), yLo + (1.0f - (s.y - o.y - pad) / (h - 2.0f * pad)) * (yHi - yLo));
    };
    dl->AddRectFilled(o, ImVec2(o.x + w, o.y + h), ImGui::GetColorU32(ImGuiCol_FrameBg), 6.0f);
    // the 0 and 1 levels, dashed: outside them = anticipation / overshoot
    for (float level : { 0.0f, 1.0f }) {
        ImVec2 a = toScreen({ 0.0f, level }), b = toScreen({ 1.0f, level });
        for (float x = a.x; x < b.x; x += 8.0f)
            dl->AddLine(ImVec2(x, a.y), ImVec2(std::min(x + 4.0f, b.x), a.y), IM_COL32(110, 110, 130, 255));
    }
    auto plot = [&](const MorphCurve& c, ImU32 col, float thick) {
        ImVec2 prev = toScreen({ 0.0f, 0.0f });
        for (int i = 1; i <= 64; ++i) {
            float x = i / 64.0f;
            ImVec2 q = toScreen({ x, c.Eval(x) });
            dl->AddLine(prev, q, col, thick);
            prev = q;
        }
    };
    int active = linked ? 0 : editing;
    plot(active == 0 ? trail : lead, (active == 0 ? colTrail : colLead) & 0x70FFFFFF, 1.5f);   // the other, faint
    MorphCurve& cur = active == 0 ? lead : trail;
    plot(cur, active == 0 ? colLead : colTrail, 2.5f);

    // handles
    glm::vec2* handles[2] = { &cur.p1, &cur.p2 };
    glm::vec2 anchors[2] = { { 0.0f, 0.0f }, { 1.0f, 1.0f } };
    for (int k = 0; k < 2; ++k) {
        ImVec2 s = toScreen(*handles[k]);
        dl->AddLine(toScreen(anchors[k]), s, IM_COL32(170, 170, 190, 200), 1.0f);
        ImGui::SetCursorScreenPos(ImVec2(s.x - 7.0f, s.y - 7.0f));
        ImGui::InvisibleButton(k == 0 ? "##h1" : "##h2", ImVec2(14.0f, 14.0f));
        bool act = ImGui::IsItemActive(), hov = ImGui::IsItemHovered();
        anyActive |= act;
        if (act && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            glm::vec2 v = fromScreen(ImGui::GetIO().MousePos);
            v.x = std::clamp(v.x, 0.0f, 1.0f);
            v.y = std::clamp(v.y, std::max(yLo, -1.0f), std::min(yHi, 2.0f));
            if (v != *handles[k]) { *handles[k] = v; changed = true; }
        }
        if (act || hov) ImGui::SetTooltip("(%.2f, %.2f)", handles[k]->x, handles[k]->y);
        dl->AddCircleFilled(s, act || hov ? 6.5f : 5.0f, act || hov ? IM_COL32(255, 220, 120, 255) : IM_COL32(245, 245, 250, 255));
    }
    dragging = anyActive;
    if (linked && changed) trail = lead;
    ImGui::SetCursorScreenPos(ImVec2(o.x, o.y + h + ImGui::GetStyle().ItemSpacing.y));
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
    ImGui::PopID();
    return changed;
}

void GradientLegend(const char* left, const char* right) {
    ImGui::TextDisabled("%s", left);
    ImGui::SameLine();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(right).x - ImGui::GetStyle().ItemSpacing.x;
    float h = ImGui::GetTextLineHeight();
    ImGui::Dummy(ImVec2(std::max(w, 20.0f), h));
    // the overlay's ramp: orange (1, 0.45, 0.15) -> blue (0.2, 0.45, 1), as display colours
    ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
        ImVec2(p.x, p.y + 0.25f * h), ImVec2(p.x + std::max(w, 20.0f), p.y + 0.75f * h),
        IM_COL32(255, 115, 38, 255), IM_COL32(51, 115, 255, 255), IM_COL32(51, 115, 255, 255), IM_COL32(255, 115, 38, 255));
    ImGui::SameLine();
    ImGui::TextDisabled("%s", right);
}
