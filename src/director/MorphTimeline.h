#pragma once
#include "MorphStyle.h"
#include <functional>
#include <string>
#include <vector>

// A pin pair: this point on the segment's start shape becomes that point on
// its end shape (both in the normalized mesh space the morph works in).
struct MorphPin {
    glm::vec3 onA = glm::vec3(0.0f);
    glm::vec3 onB = glm::vec3(0.0f);
};

// One step of the timeline: key k -> key k + 1.
struct MorphSegmentSpec {
    float duration = 3.0f;      // seconds
    int   preset = 0;           // BuiltinStyles() index the style came from (-1 = custom)
    MorphStyle style;           // how it moves (copied from the preset, then editable)
    std::vector<MorphPin> pins; // which parts must land where (shared topology only)
};

// Shapes as keyframes: key 0 -> key 1 -> ... -> key n. Every neighbouring
// pair is its own morph (its own bake), and every segment can move in its
// own style. One clock runs over the whole timeline: progress 0..1, split
// between the segments by their durations.
class MorphTimeline {
public:
    std::vector<int> keys;               // mesh indices (Window::meshPaths + 1; 0 = blob)
    std::vector<MorphSegmentSpec> segs;  // keys.size() - 1 of them
    int selected = 0;                    // the segment the Director panel edits

    MorphTimeline();

    // Keep exactly one segment per neighbouring pair of keys (new ones copy
    // the last segment's style).
    void SyncSegments();

    float Total() const;   // seconds

    // Which segment shows timeline progress p in [0, 1], and how far through
    // it that is (0..1). A key between two segments belongs to the later one.
    struct Place { int seg = 0; float local = 0.0f; };
    Place Locate(float p) const;

    // Timeline progress at the start / end of segment k.
    float SegmentStart(int k) const;
    float SegmentEnd(int k) const;

    void ApplyPreset(int seg, int preset);

    // The key strip and the selected segment's duration. shapeCombo draws one
    // shape picker (Window's list of meshes) and returns true on a change.
    // Returns true if the keys changed (a rebake is needed).
    bool DrawGUI(const std::function<bool(const char*, int&)>& shapeCombo);
};
