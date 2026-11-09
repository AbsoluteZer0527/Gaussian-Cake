#pragma once
#include "core.h"
#include "MeshTopology.h"
#include "SurfaceQuery.h"
#include <vector>

// Non-rigid template fitting: deform a template mesh onto a target surface
// while keeping the template's local shape (non-rigid ICP with an
// ARAP-rotated Laplacian regularizer; Amberg et al. 2007, Sorkine et al. 2004).
// The result has the template's connectivity and lies on the target surface,
// so the two shapes share one topology and can be morphed vertex-by-vertex.
struct FitParams {
    int   stages          = 6;      // stiffness levels, from stiff to loose
    int   itersPerStage   = 4;
    float stiffnessStart  = 50.0f;
    float stiffnessEnd    = 0.3f;
    float stiffnessScale  = 1.0f;   // UI multiplier on the whole schedule
    float normalCosMin    = 0.5f;   // reject matches whose normals differ > 60 deg
    float backwardWeight  = 1.0f;   // pull from target samples to nearest template vertex
    int   relaxRounds     = 3;      // tangential relax + reproject after the fit
    bool  cotanLaplacian  = true;   // cotan (geometry-aware) vs uniform Laplacian regularizer

    // Pins (landmarks): template vertex -> the target point it must land on,
    // e.g. "this nose goes to that spout". The shape-keeping term carries the
    // region around each pin along, so a few pins steer the whole fit.
    std::vector<std::pair<int, glm::vec3>> pins;
    float pinWeight       = 50.0f;
    float farFactor       = 0.0f;   // drop forward matches farther than this x the median (0 = keep all)
    int   untangleRounds  = 24;
    // One factorization per stage instead of per iteration: dropped matches
    // hold the vertex in place instead of letting it go. ~2.5x faster, but it
    // converges slower from far away, so only for fits that start close (the
    // Piecewise keyframes, each starting from the last).
    bool  constantWeights = false;     // smoothing passes over folded regions after the fit (0 = off)
};

struct FitStats {
    float meanDist = 0.0f;   // template-to-target distance before the final lock
    float maxDist  = 0.0f;
    int   flipped  = 0;      // faces whose normal turned > 90 deg vs. the pre-aligned template
    float pinError = 0.0f;
    int   flippedBeforeLock = 0;   // folds after the iterative fit
    int   flippedAfterLock  = 0;   // ...after snapping onto the surface (before the relax)
    int   invertedBeforeUntangle = 0;   // faces facing against the target surface, before untangling
    int   inverted = 0;            // ...at the end: the visible folds   // mean distance of the pinned vertices to their targets, before the lock
    double ms      = 0.0;
};

class MeshFitter {
public:
    // Faces whose normal turned more than 90 degrees from `rest` (folded over);
    // their indices go to flippedFaces if given.
    // Faces that face against the target surface they lie on: visible folds.
    static int CountInverted(const IndexedMesh& tmpl, const std::vector<glm::vec3>& X, const SurfaceBVH& bvh,
                             std::vector<int>* faces = nullptr);

    static int CountFlips(const IndexedMesh& tmpl, const std::vector<glm::vec3>& rest,
                          const std::vector<glm::vec3>& X, std::vector<int>* flippedFaces = nullptr);

    // tmpl:     template connectivity (positions unused)
    // start:    template positions already roughly aligned to the target
    // target:   target triangle soup
    // returns   fitted positions, one per template vertex, on the target surface
    static std::vector<glm::vec3> Fit(const IndexedMesh& tmpl,
                                      const std::vector<glm::vec3>& start,
                                      const std::vector<Triangle>& target,
                                      const FitParams& params,
                                      FitStats* stats = nullptr);
};
