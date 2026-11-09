#pragma once
#include "MorphSpring.h"
#include "MorphStyle.h"
#include "MorphTimeline.h"
#include "SurfaceQuery.h"
#include <vector>

// The Morph Director's per-point state: when each point leaves (the wave
// from an origin, plus paint), and where it is this frame (curves, springs).
//
// It is bound to a set of points that move from A to B one-to-one: the
// shared-topology template's vertices, or the splats. Evaluate turns the
// timeline into one time per point, which the renderer blends with
// (start + u (end - start)).
class MorphDirector {
public:
    // Navigate: turn the camera. StartPoint: click where the change starts.
    // TimingBrush: paint parts to change sooner or later. Pin: pair a point
    // on the start shape with where it must land on the end shape.
    enum class Tool { Navigate, StartPoint, TimingBrush, Pin };
    Tool tool = Tool::Navigate;
    bool brushLater = false;              // the brush makes parts change later (else sooner)

    // Under the mouse (the brush cursor): a point on the shown shape
    bool hasHover = false;
    glm::vec3 hoverPoint = glm::vec3(0.0f);

    glm::vec3 origin = glm::vec3(0.0f);   // the clicked origin (normalized mesh space)
    bool  hasOrigin = false;
    float brushRadius   = 0.2f;
    float brushStrength = 0.15f;

    // Pin tool: the first click (on A) waits here for its partner on B
    bool pinWaitingForB = false;
    glm::vec3 pendingA = glm::vec3(0.0f);

    // Heat wave constants (as in Splats mode): how far the source's heat
    // spreads, and how many of the points are the source.
    double sigma = 0.6, seedFraction = 0.02;

    // Bind to a new set of points (keeps the origin, drops paint and springs).
    // cellSize: the heat solve's neighbour-search cell (0 = estimate it).
    // slot: whose points these are (the timeline segment). Paint belongs to
    // its slot: binding the same slot to new points (preview -> baked
    // template) carries the paint over by position; other slots keep theirs.
    void Bind(const std::vector<glm::vec3>& points, float cellSize = 0.0f, int slot = 0);
    bool Bound() const { return !points.empty(); }
    size_t Count() const { return points.size(); }
    const std::vector<glm::vec3>& Points() const { return points; }

    // Make the delays match this source (cheap when it hasn't changed).
    void UpdateDelay(OriginMode mode);
    // Forget the delays, so the next UpdateDelay solves again.
    void InvalidateDelay() { solvedMode = -1; }

    // Timing brush: sign < 0 makes the area change sooner, > 0 later.
    void Paint(const glm::vec3& center, float sign);
    void ClearPaint();
    bool HasPaint() const { return hasPaint; }

    const std::vector<float>& Delay() const { return delay; }
    // Bumped whenever the delays change (renderers that keep a copy re-upload).
    int DelayVersion() const { return delayVersion; }

    // Each point's time at segment progress `local`, after the wave, the
    // style's curves and (dt > 0) the springs.
    const std::vector<float>& Evaluate(float local, float dt, const MorphStyle& style);
    // Put the springs at rest on this frame's targets (after a jump in time).
    void Snap(float local, const MorphStyle& style);
    bool Settled() const { return springs.Settled(); }

    // Overlay for the tools: a colour per displayed point (they need not be
    // the bound points): the delay as warm (early) to cool (late), the origin
    // and the pins as bright dots. pinsOn = the pins' ends on the shown shape.
    std::vector<glm::vec3> OverlayColors(const std::vector<glm::vec3>& displayed,
                                         const std::vector<glm::vec3>& pinsOn) const;

private:
    std::vector<glm::vec3> points;
    std::vector<float> heat, paint, delay, targets, u;
    PointGrid grid;
    float cell = 0.05f;          // neighbour-search cell for the heat solve
    int   solvedMode = -1;       // OriginMode the heat was solved for (-1 = none)
    glm::vec3 solvedOrigin = glm::vec3(1e9f);
    bool  hasPaint = false;
    int   delayVersion = 0;
    ScalarSprings springs;
    int   slot = 0;

    struct SavedPaint { std::vector<glm::vec3> points; std::vector<float> paint; };
    std::vector<std::pair<int, SavedPaint>> savedPaint;   // by slot

    // Recent heat solves, so switching back and forth between segments is free
    struct CachedHeat {
        size_t n; glm::vec3 first, middle; int mode; glm::vec3 origin;
        std::vector<float> heat;
    };
    std::vector<CachedHeat> heatCache;

    void Combine();   // delay = clamp(heat + paint)
};
