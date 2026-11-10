#pragma once
#include "core.h"
#include "MeshFitter.h"
#include "MeshTopology.h"
#include "SDFMorph.h"
#include "SurfaceColor.h"
#include <functional>
#include <string>
#include <vector>

// Topology morph across a change of genus, holes or parts, with real meshes
// all the way: "piecewise shared topology".
//
// The OT-aligned SDF morph decides *where* the shape goes, and its
// in-between surfaces show *when* the topology changes (a hole opens, two
// parts merge). Between two such events the topology is fixed, so one mesh
// can follow the surface: it is fitted (MeshFitter) onto the SDF surface at
// a row of keyframe times, each fit starting from the last. The first
// segment's mesh is A itself, the last one's is B itself, so both ends are
// the real meshes; a segment in between uses the SDF surface at its middle.
// At each event the two neighbouring meshes lie on (almost) the same
// surface, and the morph switches from one to the other: the topology change
// happens there, at one visible moment.
struct PiecewiseSegment {
    float tStart = 0.0f, tEnd = 1.0f;     // the part of the timeline this mesh covers
    std::vector<unsigned int> indices;    // the mesh's triangles
    std::vector<float> times;             // keyframe times, ascending, inside [tStart, tEnd]
    std::vector<std::vector<glm::vec3>> positions;   // per keyframe, one position per vertex
    std::vector<std::vector<glm::vec3>> colors;      // per keyframe (empty: uncoloured)
};

struct PiecewiseEvent {
    float t = 0.0f;            // where the switch happens
    std::string what;          // e.g. "genus 0 -> 1"
};

class PiecewiseMorph {
public:
    struct Params {
        int       keyframes = 12;   // fits per unit of time
        FitParams fit;
        // Progress (called on the building thread): phase 0 = finding the
        // topology changes, 1 = fitting the meshes; done of total.
        std::function<void(int phase, int done, int total)> progress;
    };

    // a / b: the cleaned end meshes (A's and B's vertices); colorsA / colorsB:
    // their vertex colours (empty: uncoloured); colorA / colorB: the meshes'
    // colour lookups (null: uncoloured). Returns false if a fit failed.
    bool Build(const SDFMorph& sdf, const IndexedMesh& a, const IndexedMesh& b,
               const std::vector<glm::vec3>& colorsA, const std::vector<glm::vec3>& colorsB,
               const SurfaceColor* colorA, const SurfaceColor* colorB, const Params& params);

    // Which segment shows time t, and that segment's surface at t.
    int  SegmentAt(float t) const;
    void Evaluate(int segment, float t, std::vector<glm::vec3>& positions,
                  std::vector<glm::vec3>& colors) const;

    // Topology of the solid inside the t-surface, read off the grid: parts,
    // genus and sealed cavities, ignoring specks (pieces and bubbles of less
    // than 0.5% of the solid's grid points).
    struct Topology {
        int parts = 0, genus = 0, cavities = 0;
        bool operator==(const Topology& o) const {
            return parts == o.parts && genus == o.genus && cavities == o.cavities;
        }
        bool operator!=(const Topology& o) const { return !(*this == o); }
    };
    static Topology SolidTopology(const SDFMorph& sdf, float t);

    std::vector<PiecewiseSegment> segments;
    std::vector<PiecewiseEvent>   events;
    int    fits = 0;
    double ms = 0.0;
};
