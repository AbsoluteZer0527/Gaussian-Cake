#pragma once
#include "core.h"
#include "OBJLoader.h"
#include "OptimalTransport.h"
#include "SurfaceColor.h"
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

// A colour at every point of the same kind of grid (the colour of the mesh's
// closest surface point), so in-between surfaces can be coloured.
struct ColorGrid {
    int n = 0;
    glm::vec3 lo = glm::vec3(0.0f);
    float h = 1.0f;
    std::vector<glm::vec3> c;    // x fastest, then y, then z

    glm::vec3 At(int i, int j, int k) const { return c[((size_t)k * n + j) * n + i]; }
    glm::vec3 Sample(const glm::vec3& p) const;   // trilinear, clamped to the grid
};

// The OT map T sampled on a grid, for maps that aren't affine (the
// Gaussian-mixture map): T(u) and its Jacobian J(u) at every grid point,
// trilinear in between. Outside the grid it continues linearly
// (T(c) + J(c)(u - c), c = the closest grid point).
struct WarpGrid {
    int n = 0;
    glm::vec3 lo = glm::vec3(0.0f);
    float h = 1.0f;
    std::vector<glm::vec3> T;     // x fastest, then y, then z
    std::vector<glm::mat3> J;

    bool Empty() const { return T.empty(); }
    glm::vec3 Sample(const glm::vec3& u, glm::mat3* outJ = nullptr) const;
};

// OT-aligned distance-field metamorphosis (Cohen-Or et al. 1998). The surface
// at time t is the zero set of
//   phi_t(y) = (1 - t) phi_A(u) + t phi_B(T(u)),   u = M_t^-1(y),
// where T is the Gaussian OT map A -> B and M_t = (1 - t) Id + t T. The shape
// travels along the OT path while the surface blends. It works for any change
// of genus or number of parts, since nothing about connectivity is kept.
//
// With a mixture map (Build's `mixture`), T is no longer affine: it is the
// Gaussian-mixture OT map (Delon & Desolneux), which lines up the shapes part
// by part instead of only as a whole. Then u = M_t^-1(y) has no closed form;
// it's found by Newton's method on M_t(u) = (1 - t) u + t T(u) = y, starting
// from the affine answer. Everything else (the blend, the exact normals by
// pullback, the colours) is the same with J(u) in place of A.
class SDFMorph {
public:
    // Builds both distance fields. resolution = grid points per axis.
    // colorA / colorB: the meshes' colours, or null for an uncoloured morph.
    // mixture: the mixture OT map to align by (null = the affine `map` only).
    void Build(const std::vector<Triangle>& trisA, const std::vector<Triangle>& trisB,
               const OTMap& map, int resolution,
               const SurfaceColor* colorA = nullptr, const SurfaceColor* colorB = nullptr,
               const MixtureOTMap* mixture = nullptr);

    // Where y on the t-surface came from on A (u = M_t^-1(y)), with T(u)
    // and J(u) (the affine map's when there is no warp).
    glm::vec3 Preimage(const glm::vec3& y, float t, glm::vec3* outTu = nullptr,
                       glm::mat3* outJ = nullptr) const;

    // How well T lines A up with B: mean |phi_B(T(x))| over points x on A's
    // surface (0 = A's surface lands exactly on B's), for the affine map and
    // for the map the morph uses.
    float AlignmentError(const std::vector<glm::vec3>& pointsOnA, bool affineOnly) const;

    bool HasWarp() const { return !warp.Empty(); }
    const WarpGrid& Warp() const { return warp; }
    float warpStrength = 0.0f;     // lambda: how much of the part-wise correction is used (0..1)

    // Extracts the t-surface as an indexed mesh (naive surface nets). With
    // outColors, also a colour per vertex: A's and B's colour at the points
    // the vertex came from, blended by t (only if the morph has colours).
    // stride > 1 samples every stride-th grid point (a coarser surface).
    void Extract(float t, std::vector<glm::vec3>& outVerts,
                 std::vector<unsigned int>& outIndices,
                 std::vector<glm::vec3>* outColors = nullptr, int stride = 1) const;

    // phi_t at the grid points (every stride-th one): outN per axis, point
    // (i, j, k) at outLo + outH (i, j, k), x fastest.
    void Field(float t, int stride, int& outN, glm::vec3& outLo, float& outH,
               std::vector<float>& out) const;

    bool   IsBuilt() const { return built; }
    bool   HasColor() const { return !colorGridA.c.empty(); }
    const SDFGrid&   GridA() const { return gridA; }
    const SDFGrid&   GridB() const { return gridB; }
    const ColorGrid& ColorGridA() const { return colorGridA; }
    const ColorGrid& ColorGridB() const { return colorGridB; }
    const glm::mat3& MapMatrix() const { return A; }
    const glm::vec3& MapOffset() const { return b; }
    double buildMs = 0.0;
    mutable double lastExtractMs = 0.0;

private:
    SDFGrid gridA, gridB;
    ColorGrid colorGridA, colorGridB;   // empty when the morph has no colours
    WarpGrid warp;                      // empty: T is the affine map below
    glm::mat3 A = glm::mat3(1.0f);   // T(x) = A x + b
    glm::vec3 b = glm::vec3(0.0f);
    bool built = false;
};
