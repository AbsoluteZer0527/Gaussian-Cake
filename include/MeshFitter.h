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
};

struct FitStats {
    float meanDist = 0.0f;   // template-to-target distance before the final lock
    float maxDist  = 0.0f;
    int   flipped  = 0;      // faces whose normal turned > 90 deg vs. the pre-aligned template
    double ms      = 0.0;
};

class MeshFitter {
public:
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
