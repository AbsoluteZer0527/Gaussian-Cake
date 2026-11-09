#pragma once
#include "MorphStyle.h"
#include "MorphTimeline.h"
#include "imgui.h"
#include <string>
#include <vector>

// The Animation window's widgets: drawn icons (the default font has no icon
// glyphs), a dope-sheet timeline track, style tiles and an easing editor.
// Pure ImGui: they draw, and report what the user did; Window applies it.

enum class Icon {
    Play, Pause, ToStart, ToEnd, Loop,
    Navigate, StartPoint, Brush, Pin,
    Plus, Clear, Link,
    SpreadAll, SpreadUp, SpreadDown, SpreadPoint,
    Sooner, Later,
};

void DrawIcon(ImDrawList* dl, Icon icon, ImVec2 center, float size, ImU32 col);

// A square button showing an icon; `active` draws it in the accent colour.
// tooltip: shown on hover (name, then shortcut / description).
bool IconButton(const char* id, Icon icon, bool active, const char* tooltip, float size = 26.0f);

// Joined icon buttons, one of them on (Figma's segmented control).
struct SegmentOption { Icon icon; const char* tooltip; };
bool IconSegmented(const char* id, int& value, const std::vector<SegmentOption>& options, float size = 26.0f);

// The accent colours
ImU32 AccentColor();              // pink: the playhead, active buttons
ImU32 StyleColor(const MorphStyle& style, float alpha = 1.0f);

// What the user did on the timeline track this frame.
struct TrackResult {
    bool  seek = false;      float progress = 0.0f;   // scrub to this timeline progress
    int   selectSeg = -1;                             // a clip was clicked
    int   selectKey = -1;                             // a key diamond was clicked
    int   retimeKey = -1;    float retimeTime = 0.0f; // a key was dragged to this time (s)
    bool  addKey = false;                             // the + button
    int   deleteKey = -1;                             // context menu
    int   shapeKey = -1, shapeMesh = -1;              // context menu: key -> mesh index
    int   styleSeg = -1, styleIndex = -1;             // context menu: clip -> preset
};

// The dope sheet: a ruler, one clip per step (coloured by its style), key
// diamonds between them and a playhead. Click the ruler to scrub, a clip to
// select it, a diamond to select that key; drag an inner diamond to retime;
// right-click for a menu. meshNames[i] names mesh index i + 1 (0 = blob).
// canScrub: something is baked (else clicks on the ruler only explain that).
void TimelineTrack(const char* id, const MorphTimeline& tl, float progress, bool editKeys, bool canScrub,
                   const std::vector<std::string>& meshNames, TrackResult& out);

// A tile showing a style's timing as a little curve, its name below.
bool StyleTile(const char* id, const MorphStyle& style, bool selected, float size);

// One canvas, two curves: Lead (the parts the change reaches first, pink)
// and Trail (the last ones, blue). `editing` picks whose handles move;
// `linked` keeps them the same. now: where the playback is (0..1, -1 =
// none), shown as a dot riding the curve. leadDefault / trailDefault: what a
// double-click resets to (null = no reset). Returns true if a curve changed.
bool EasingEditor(const char* id, MorphCurve& lead, MorphCurve& trail, bool& linked, int& editing,
                  float now = -1.0f, const MorphCurve* leadDefault = nullptr,
                  const MorphCurve* trailDefault = nullptr);

// A small orange -> blue bar, the colours of the timing overlay.
void GradientLegend(const char* left, const char* right);

// Where the interactive parts were drawn last frame (screen rects), so a
// scripted run (menv --uitest) can click and drag them like a user would.
namespace UiDebug {
void Record(const std::string& name, ImVec2 min, ImVec2 max);
bool Get(const std::string& name, ImVec2& min, ImVec2& max);
}

// A rounded label with a coloured background (counts, status); tooltip on hover.
void Badge(const char* text, ImU32 bg, const char* tooltip = nullptr);
