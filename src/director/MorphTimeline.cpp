#include "MorphTimeline.h"
#include "imgui.h"
#include <algorithm>

MorphTimeline::MorphTimeline() {
    keys = { 0, 0 };
    SyncSegments();
    ApplyPreset(0, 0);
}

void MorphTimeline::SyncSegments() {
    if (keys.size() < 2) keys.resize(2, keys.empty() ? 0 : keys.back());
    size_t want = keys.size() - 1;
    while (segs.size() < want) segs.push_back(segs.empty() ? MorphSegmentSpec() : segs.back());
    segs.resize(want);
    selected = std::clamp(selected, 0, (int)segs.size() - 1);
}

float MorphTimeline::Total() const {
    float t = 0.0f;
    for (const auto& s : segs) t += std::max(s.duration, 0.01f);
    return t;
}

float MorphTimeline::SegmentStart(int k) const {
    float total = Total(), t = 0.0f;
    for (int i = 0; i < k && i < (int)segs.size(); ++i) t += std::max(segs[i].duration, 0.01f);
    return total > 0.0f ? t / total : 0.0f;
}

float MorphTimeline::SegmentEnd(int k) const {
    if (k >= (int)segs.size() - 1) return 1.0f;
    return SegmentStart(k + 1);
}

MorphTimeline::Place MorphTimeline::Locate(float p) const {
    Place place;
    if (segs.empty()) return place;
    float total = Total();
    float time = std::clamp(p, 0.0f, 1.0f) * total;
    float start = 0.0f;
    for (int k = 0; k < (int)segs.size(); ++k) {
        float d = std::max(segs[k].duration, 0.01f);
        bool last = k == (int)segs.size() - 1;
        if (time < start + d || last) {
            place.seg = k;
            place.local = std::clamp((time - start) / d, 0.0f, 1.0f);
            if (last && p >= 1.0f) place.local = 1.0f;
            return place;
        }
        start += d;
    }
    return place;
}

void MorphTimeline::ApplyPreset(int seg, int preset) {
    if (seg < 0 || seg >= (int)segs.size()) return;
    const auto& styles = BuiltinStyles();
    if (preset < 0 || preset >= (int)styles.size()) return;
    segs[seg].style = styles[preset];
    segs[seg].preset = preset;
}

bool MorphTimeline::DrawGUI(const std::function<bool(const char*, int&)>& shapeCombo) {
    bool changed = false, reordered = false;
    int removeKey = -1, moveUp = -1;
    for (int k = 0; k < (int)keys.size(); ++k) {
        ImGui::PushID(k);
        char label[32];
        snprintf(label, sizeof(label), k == 0 ? "Start" : "Key %d", k);
        if (shapeCombo(label, keys[k])) {
            changed = true;   // pins were placed on the old shape
            if (k > 0) segs[k - 1].pins.clear();
            if (k < (int)segs.size()) segs[k].pins.clear();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(k == 0);
        if (ImGui::SmallButton("^")) moveUp = k;
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Move this shape earlier");
        ImGui::SameLine();
        ImGui::BeginDisabled(keys.size() <= 2);
        if (ImGui::SmallButton("x")) removeKey = k;
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Remove this shape");

        // the segment from this key to the next one: a selectable row
        if (k + 1 < (int)keys.size()) {
            const MorphSegmentSpec& s = segs[k];
            char row[96];
            snprintf(row, sizeof(row), "   |  %s, %.1f s%s", s.style.name.c_str(), s.duration,
                     s.pins.empty() ? "" : (", " + std::to_string(s.pins.size()) + " pins").c_str());
            if (ImGui::Selectable(row, selected == k)) selected = k;
            ImGui::SetItemTooltip("Click to edit how this step moves");
        }
        ImGui::PopID();
    }
    if (ImGui::Button("+ Add shape")) {
        keys.push_back(keys.back());
        SyncSegments();
        selected = (int)segs.size() - 1;
        segs.back().pins.clear();
        changed = true;
    }
    if (moveUp > 0) {
        std::swap(keys[moveUp], keys[moveUp - 1]);
        changed = reordered = true;
    }
    if (removeKey >= 0) {
        keys.erase(keys.begin() + removeKey);
        int seg = std::min(removeKey, (int)segs.size() - 1);
        segs.erase(segs.begin() + seg);
        SyncSegments();
        changed = reordered = true;
    }
    if (reordered)   // pins were placed on the old pairs of shapes
        for (auto& s : segs) s.pins.clear();
    return changed;
}
