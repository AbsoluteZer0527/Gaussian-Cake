#pragma once
#include "MorphStyle.h"
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
    bool easingLinked = true;   // one easing curve for all parts (Lead = Trail)
    int  easingEdit = 0;        // the curve the easing editor's handles move (0 Lead, 1 Trail)
};

// Shapes as keyframes: key 0 -> key 1 -> ... -> key n. Every neighbouring
// pair is its own morph (its own bake), and every segment can move in its
// own style. One clock runs over the whole timeline: progress 0..1, split
// between the segments by their durations.
class MorphTimeline {
public:
    std::vector<int> keys;               // mesh indices (Window::meshPaths + 1; 0 = blob)
    std::vector<MorphSegmentSpec> segs;  // keys.size() - 1 of them
    int selected = 0;                    // the step the Animation window edits
    int selectedKey = -1;                // a key diamond clicked on the track (-1 = none)

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

    // Time of key k, in seconds from the start.
    float KeyTime(int k) const;

    // Editing the keys (each clears the pins that no longer fit).
    void AddKey(int mesh);                 // at the end, a copy of the last step's style
    void RemoveKey(int k);                 // the steps on both sides merge into one
    void SetKey(int k, int mesh);          // a different shape for key k
    // Move inner key k to time t (s): the steps on both sides change length,
    // their total stays, and each keeps at least kMinStep seconds.
    void Retime(int k, float t);
    static constexpr float kMinStep = 0.3f;
};
