#pragma once
#include "core.h"
#include "OBJLoader.h"
#include "OptimalTransport.h"
#include <vector>

// Signed distance values on a regular n^3 grid of points.
struct SDFGrid {
    int n = 0;
    glm::vec3 lo = glm::vec3(0.0f);
    float h = 1.0f;              // spacing between grid points
    std::vector<float> v;        // x fastest, then y, then z

    float At(int i, int j, int k) const { return v[((size_t)k * n + j) * n + i]; }

    // Trilinear lookup; outside the grid, the boundary value plus the distance
    // to the grid box (the boundary is always outside the shape).
    float Sample(const glm::vec3& p) const;
};

// OT-aligned distance-field metamorphosis (Cohen-Or et al. 1998). The surface
// at time t is the zero set of
//   phi_t(y) = (1 - t) phi_A(u) + t phi_B(T(u)),   u = M_t^-1(y),
// where T is the Gaussian OT map A -> B and M_t = (1 - t) Id + t T. The shape
// travels along the OT path while the surface blends. It works for any change
// of genus or number of parts, since nothing about connectivity is kept.
class SDFMorph {
public:
    // Builds both distance fields. resolution = grid points per axis.
    void Build(const std::vector<Triangle>& trisA, const std::vector<Triangle>& trisB,
               const OTMap& map, int resolution);

    // Extracts the t-surface as an indexed mesh (naive surface nets).
    void Extract(float t, std::vector<glm::vec3>& outVerts,
                 std::vector<unsigned int>& outIndices) const;

    bool   IsBuilt() const { return built; }
    double buildMs = 0.0;
    mutable double lastExtractMs = 0.0;

private:
    SDFGrid gridA, gridB;
    glm::mat3 A = glm::mat3(1.0f);   // T(x) = A x + b
    glm::vec3 b = glm::vec3(0.0f);
    bool built = false;
};
