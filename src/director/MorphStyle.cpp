#include "MorphStyle.h"
#include "SplatHeat.h"
#include <algorithm>
#include <cctype>

const std::vector<MorphStyle>& BuiltinStyles() {
    static const std::vector<MorphStyle> styles = [] {
        std::vector<MorphStyle> list;
        MorphStyle s;

        // Classic: the morph as it was before the Director (ease in-out, together)
        s = MorphStyle();
        s.name = "Classic";
        s.color = glm::vec3(0.55f, 0.6f, 0.72f);
        s.origin = OriginMode::None;
        s.early = s.late = MorphCurve(0.45f, 0.0f, 0.55f, 1.0f);
        s.hint = "Every part changes at once, easing in and out";
        list.push_back(s);

        s = MorphStyle();
        s.name = "Bake";
        s.color = glm::vec3(0.95f, 0.62f, 0.3f);
        s.origin = OriginMode::Bottom;
        s.wave = 0.45f;
        s.early = s.late = MorphCurve(0.42f, 0.0f, 0.58f, 1.0f);
        s.hint = "Rises from the bottom like a cake in the oven";
        list.push_back(s);

        s = MorphStyle();
        s.name = "Ripple";
        s.color = glm::vec3(0.3f, 0.75f, 0.92f);
        s.origin = OriginMode::Clicked;
        s.wave = 0.55f;
        s.early = MorphCurve(0.2f, 0.0f, 0.3f, 1.0f);
        s.late  = MorphCurve(0.2f, 0.0f, 0.3f, 1.5f);   // the last parts overshoot ~10%
        s.jiggle = 0.3f;
        s.cohesion = 0.4f;
        s.midGrow = 0.2f;
        s.hint = "A ring rolls out from the clicked spot (Set Origin tool)";
        list.push_back(s);

        s = MorphStyle();
        s.name = "Melt";
        s.color = glm::vec3(0.72f, 0.48f, 0.88f);
        s.origin = OriginMode::Top;
        s.wave = 0.35f;
        s.early = MorphCurve(0.6f, 0.0f, 0.9f, 1.0f);
        s.late  = MorphCurve(0.7f, 0.0f, 0.95f, 1.0f);
        s.sag = 0.35f;
        s.cohesion = 0.8f;
        s.midGrow = 0.5f;
        s.hint = "Droops from the top, then sets into the new shape";
        list.push_back(s);

        s = MorphStyle();
        s.name = "Pop";
        s.color = glm::vec3(1.0f, 0.45f, 0.7f);
        s.origin = OriginMode::None;
        s.early = s.late = MorphCurve(0.3f, -0.5f, 0.55f, 1.8f);   // dips to -0.07, peaks at 1.17
        s.jiggle = 0.7f;
        s.midGrow = 0.1f;
        s.hint = "All at once: a pull back, an overshoot and a wobble";
        list.push_back(s);

        s = MorphStyle();
        s.name = "Dissolve";
        s.color = glm::vec3(0.42f, 0.8f, 0.55f);
        s.origin = OriginMode::Bottom;
        s.wave = 0.5f;
        s.early = s.late = MorphCurve(0.5f, 0.0f, 0.5f, 1.0f);
        s.cohesion = 0.0f;
        s.midGrow = 0.8f;
        s.splatsBest = true;
        s.hint = "Breaks into drifting specks that gather again (best in Splats)";
        list.push_back(s);
        return list;
    }();
    return styles;
}

int FindStyle(const std::string& name) {
    auto lower = [](std::string s) {
        for (char& c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    const auto& styles = BuiltinStyles();
    for (size_t i = 0; i < styles.size(); ++i)
        if (lower(styles[i].name) == lower(name)) return (int)i;
    return -1;
}

float VertexTime(float t, float wave, float delay, const MorphCurve& early, const MorphCurve& late) {
    float local = WaveTime(t, wave, delay);
    float d = std::clamp(delay, 0.0f, 1.0f);
    return (1.0f - d) * early.Eval(local) + d * late.Eval(local);
}
