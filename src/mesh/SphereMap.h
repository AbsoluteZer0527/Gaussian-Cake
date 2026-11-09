#pragma once
#include "core.h"
#include "MeshTopology.h"
#include <utility>
#include <vector>

// Fold-free matching between two closed genus-0 meshes, through the sphere.
//
// Each mesh is flowed onto the unit sphere (conformalized mean-curvature
// flow, Kazhdan, Solomon & Ben-Chen 2012), any triangles left inside out are
// relaxed, and the area is spread out by a Moebius centring. Both spheres are
// then turned to agree with the shapes' own rough alignment, and every
// template vertex is carried, by its sphere position, onto the target's real
// surface. Both sphere maps are one-to-one, so the correspondence has no
// folds, and every vertex lies exactly on the target.
struct SphereStats {
    int    flowSteps = 0;
    float  roundness = 0.0f;   // (max - min) / mean distance from the centre, before projecting
    int    flippedAfterFlow = 0;
    int    flipped = 0;        // inside-out sphere triangles at the end (0 = one-to-one)
    double ms = 0.0;
};

namespace SphereMap {

// The mesh's vertices on the unit sphere. false if it isn't one closed
// genus-0 part, or triangles stay inside out (then don't use the result).
bool Compute(const IndexedMesh& m, std::vector<glm::vec3>& onSphere, SphereStats* stats = nullptr);

// Inside-out triangles of a sphere embedding (normal pointing at the centre).
int CountFlipped(const IndexedMesh& m, const std::vector<glm::vec3>& onSphere);

// The rotation R that best takes the template's sphere onto the target's:
// pairs[i] = (template sphere point, target sphere point) with weights.
glm::mat3 Kabsch(const std::vector<std::pair<glm::vec3, glm::vec3>>& pairs, const std::vector<float>& weights);

// A Moebius map of the sphere (one-to-one, no folds) sending from[k] to
// to[k] exactly, for up to 3 pairs; with fewer, extra points are held fixed.
// Applies it to every point of `points`.
void MoebiusFit(const std::vector<glm::vec3>& from, const std::vector<glm::vec3>& to,
                std::vector<glm::vec3>& points);

// The target's sphere point under a point of its real surface.
glm::vec3 SpherePointOf(const IndexedMesh& target, const std::vector<glm::vec3>& targetSphere,
                        const glm::vec3& onSurface);

// Every template sphere point (already rotated) carried onto the target's
// real surface: the target sphere triangle it falls in, the same
// barycentric weights on the target's own triangle.
std::vector<glm::vec3> Transfer(const std::vector<glm::vec3>& points, const IndexedMesh& target,
                                const std::vector<glm::vec3>& targetSphere);

}  // namespace SphereMap
