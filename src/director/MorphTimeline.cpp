#include "MorphTimeline.h"
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
    segs[seg].basePreset = preset;
    segs[seg].easingLinked = styles[preset].early == styles[preset].late;
    segs[seg].easingEdit = 0;
}

float MorphTimeline::KeyTime(int k) const {
    float t = 0.0f;
    for (int i = 0; i < k && i < (int)segs.size(); ++i) t += std::max(segs[i].duration, 0.01f);
    return t;
}

void MorphTimeline::AddKey(int mesh) {
    keys.push_back(mesh);
    SyncSegments();
    segs.back().pins.clear();
    selected = (int)segs.size() - 1;
}

void MorphTimeline::RemoveKey(int k) {
    if (keys.size() <= 2 || k < 0 || k >= (int)keys.size()) return;
    // the step before the key absorbs the one after it (the first key: the next step goes)
    int keep = std::max(k - 1, 0), drop = std::min(k, (int)segs.size() - 1);
    if (k > 0 && k < (int)keys.size() - 1) segs[keep].duration += segs[drop].duration;
    keys.erase(keys.begin() + k);
    segs.erase(segs.begin() + drop);
    if (keep < (int)segs.size()) segs[keep].pins.clear();
    SyncSegments();
    selectedKey = -1;
}

void MorphTimeline::SetKey(int k, int mesh) {
    if (k < 0 || k >= (int)keys.size() || keys[k] == mesh) return;
    keys[k] = mesh;
    // pins were placed on the old shape
    if (k > 0) segs[k - 1].pins.clear();
    if (k < (int)segs.size()) segs[k].pins.clear();
}

void MorphTimeline::Retime(int k, float t) {
    if (k <= 0 || k >= (int)keys.size() - 1) return;
    float a = KeyTime(k - 1), b = KeyTime(k + 1);
    if (b - a < 2.0f * kMinStep) return;
    t = std::clamp(t, a + kMinStep, b - kMinStep);
    segs[k - 1].duration = t - a;
    segs[k].duration = b - t;
}
