#include "SDFMorph.h"
#include "SurfaceQuery.h"
#include "Parallel.h"

#include <chrono>
#include <cmath>
#include <iostream>

// All meshes are normalized to [-1, 1]^3; pad so the surface never touches
// the grid boundary.
static const float kDomain = 1.3f;

float SDFGrid::Sample(const glm::vec3& p) const {
    glm::vec3 g = (p - lo) / h;
    glm::vec3 gc = glm::clamp(g, glm::vec3(0.0f), glm::vec3((float)(n - 1) - 1e-4f));
    float outside = h * glm::length(g - gc);

    int i = (int)gc.x, j = (int)gc.y, k = (int)gc.z;
    float fx = gc.x - i, fy = gc.y - j, fz = gc.z - k;
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
    float c00 = lerp(At(i, j,     k),     At(i + 1, j,     k),     fx);
    float c10 = lerp(At(i, j + 1, k),     At(i + 1, j + 1, k),     fx);
    float c01 = lerp(At(i, j,     k + 1), At(i + 1, j,     k + 1), fx);
    float c11 = lerp(At(i, j + 1, k + 1), At(i + 1, j + 1, k + 1), fx);
    return lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz) + outside;
}

glm::vec3 ColorGrid::Sample(const glm::vec3& p) const {
    glm::vec3 g = glm::clamp((p - lo) / h, glm::vec3(0.0f), glm::vec3((float)(n - 1) - 1e-4f));
    int i = (int)g.x, j = (int)g.y, k = (int)g.z;
    glm::vec3 f = g - glm::vec3((float)i, (float)j, (float)k);
    glm::vec3 c00 = glm::mix(At(i, j,     k),     At(i + 1, j,     k),     f.x);
    glm::vec3 c10 = glm::mix(At(i, j + 1, k),     At(i + 1, j + 1, k),     f.x);
    glm::vec3 c01 = glm::mix(At(i, j,     k + 1), At(i + 1, j,     k + 1), f.x);
    glm::vec3 c11 = glm::mix(At(i, j + 1, k + 1), At(i + 1, j + 1, k + 1), f.x);
    return glm::mix(glm::mix(c00, c10, f.y), glm::mix(c01, c11, f.y), f.z);
}

static ColorGrid BuildColorGrid(const SurfaceColor& color, const SDFGrid& like) {
    ColorGrid g;
    g.n = like.n;
    g.lo = like.lo;
    g.h = like.h;
    const int n = g.n;
    g.c.assign((size_t)n * n * n, glm::vec3(1.0f));
    ParallelFor(n * n * n, [&](int idx) {
        int i = idx % n, j = (idx / n) % n, k = idx / (n * n);
        g.c[idx] = color.At(g.lo + g.h * glm::vec3((float)i, (float)j, (float)k));
    });
    return g;
}

// Signed distance of one mesh on the grid. Inside/outside comes from the
// generalized winding number. Where the winding number is ambiguous (near
// open sheets and holes, |w| ~ 0.5) the surface also gets a thin shell, so an
// open sheet still becomes a thin solid instead of disappearing.
static SDFGrid BuildGrid(const std::vector<Triangle>& tris, int n) {
    SDFGrid g;
    g.n = n;
    g.lo = glm::vec3(-kDomain);
    g.h = 2.0f * kDomain / (float)(n - 1);
    g.v.assign((size_t)n * n * n, 0.0f);

    SurfaceBVH bvh;
    bvh.Build(tris);
    const float shell = 1.0f * g.h;   // open sheets become ~2 grid cells thick

    ParallelFor(n * n * n, [&](int idx) {
        int i = idx % n, j = (idx / n) % n, k = idx / (n * n);
        glm::vec3 p = g.lo + g.h * glm::vec3((float)i, (float)j, (float)k);
        float d = std::sqrt(bvh.Closest(p).dist2);
        float w = std::abs(bvh.WindingNumber(p));
        float s = (w >= 0.5f) ? -d : d;
        float ambiguity = glm::clamp(1.0f - std::abs(2.0f * w - 1.0f), 0.0f, 1.0f);
        g.v[idx] = std::min(s, d - shell * ambiguity);
    });
    return g;
}

void SDFMorph::Build(const std::vector<Triangle>& trisA, const std::vector<Triangle>& trisB,
                     const OTMap& map, int resolution,
                     const SurfaceColor* colorA, const SurfaceColor* colorB) {
    auto t0 = std::chrono::high_resolution_clock::now();
    resolution = std::max(16, resolution);
    gridA = BuildGrid(trisA, resolution);
    gridB = BuildGrid(trisB, resolution);
    colorGridA = ColorGrid();
    colorGridB = ColorGrid();
    if (colorA && colorB) {
        colorGridA = BuildColorGrid(*colorA, gridA);
        colorGridB = BuildColorGrid(*colorB, gridB);
    }

    // glm is column-major: A[c][r]
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A[c][r] = (float)map.A(r, c);
    Eigen::Vector3d be = map.muB - map.A * map.muA;
    b = glm::vec3((float)be.x(), (float)be.y(), (float)be.z());

    built = true;
    buildMs = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t0).count();
}

void SDFMorph::Extract(float t, std::vector<glm::vec3>& outVerts,
                       std::vector<unsigned int>& outIndices,
                       std::vector<glm::vec3>* outColors) const {
    outVerts.clear();
    outIndices.clear();
    if (outColors) outColors->clear();
    if (!built) return;
    auto t0 = std::chrono::high_resolution_clock::now();

    t = glm::clamp(t, 0.0f, 1.0f);
    const int n = gridA.n;
    const glm::vec3 lo = gridA.lo;
    const float h = gridA.h;

    // M_t(x) = L_t x + t b,  L_t = (1 - t) I + t A  (A is SPD, so L_t is invertible)
    glm::mat3 Lt = (1.0f - t) * glm::mat3(1.0f) + t * A;
    glm::mat3 LtInv = glm::inverse(Lt);
    glm::vec3 shift = t * b;

    std::vector<float> f((size_t)n * n * n);
    ParallelFor(n * n * n, [&](int idx) {
        int i = idx % n, j = (idx / n) % n, k = idx / (n * n);
        glm::vec3 y = lo + h * glm::vec3((float)i, (float)j, (float)k);
        glm::vec3 u = LtInv * (y - shift);
        float fa = (t < 1.0f) ? gridA.Sample(u) : 0.0f;
        float fb = (t > 0.0f) ? gridB.Sample(A * u + b) : 0.0f;
        f[idx] = (1.0f - t) * fa + t * fb;
    });
    auto F = [&](int i, int j, int k) { return f[((size_t)k * n + j) * n + i]; };

    // Naive surface nets (Gibson 1998): one vertex per cell the surface passes
    // through, at the mean of the edge crossings in that cell.
    const int c = n - 1;   // cells per axis
    auto cellId = [&](int i, int j, int k) { return ((size_t)k * c + j) * c + i; };
    std::vector<int> cellVert((size_t)c * c * c, -1);

    static const int corner[8][3] = {
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1} };
    static const int edge[12][2] = {
        {0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7} };

    for (int k = 0; k < c; ++k)
        for (int j = 0; j < c; ++j)
            for (int i = 0; i < c; ++i) {
                float val[8];
                int inside = 0;
                for (int q = 0; q < 8; ++q) {
                    val[q] = F(i + corner[q][0], j + corner[q][1], k + corner[q][2]);
                    if (val[q] < 0.0f) ++inside;
                }
                if (inside == 0 || inside == 8) continue;

                glm::vec3 sum(0.0f);
                int cnt = 0;
                for (const auto& e : edge) {
                    float a = val[e[0]], bb = val[e[1]];
                    if ((a < 0.0f) == (bb < 0.0f)) continue;
                    float s = a / (a - bb);
                    glm::vec3 pa((float)corner[e[0]][0], (float)corner[e[0]][1], (float)corner[e[0]][2]);
                    glm::vec3 pb((float)corner[e[1]][0], (float)corner[e[1]][1], (float)corner[e[1]][2]);
                    sum += pa + s * (pb - pa);
                    ++cnt;
                }
                glm::vec3 local = sum / (float)cnt;
                cellVert[cellId(i, j, k)] = (int)outVerts.size();
                outVerts.push_back(lo + h * (glm::vec3((float)i, (float)j, (float)k) + local));
            }

    // One quad per grid edge with a sign change, joining the 4 cells around
    // that edge. Wound so the normal points from inside (phi < 0) to outside.
    auto quad = [&](size_t q0, size_t q1, size_t q2, size_t q3, bool flip) {
        int v[4] = { cellVert[q0], cellVert[q1], cellVert[q2], cellVert[q3] };
        if (v[0] < 0 || v[1] < 0 || v[2] < 0 || v[3] < 0) return;
        if (flip) std::swap(v[1], v[3]);
        outIndices.insert(outIndices.end(), { (unsigned)v[0], (unsigned)v[1], (unsigned)v[2],
                                              (unsigned)v[0], (unsigned)v[2], (unsigned)v[3] });
    };

    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                bool in0 = F(i, j, k) < 0.0f;
                // x-edge: cells in the (y, z) plane, CCW about +x
                if (i + 1 < n && j >= 1 && k >= 1 && j < c && k < c && in0 != (F(i + 1, j, k) < 0.0f))
                    quad(cellId(i, j - 1, k - 1), cellId(i, j, k - 1), cellId(i, j, k), cellId(i, j - 1, k), !in0);
                // y-edge: cells in the (z, x) plane, CCW about +y
                if (j + 1 < n && i >= 1 && k >= 1 && i < c && k < c && in0 != (F(i, j + 1, k) < 0.0f))
                    quad(cellId(i - 1, j, k - 1), cellId(i - 1, j, k), cellId(i, j, k), cellId(i, j, k - 1), !in0);
                // z-edge: cells in the (x, y) plane, CCW about +z
                if (k + 1 < n && i >= 1 && j >= 1 && i < c && j < c && in0 != (F(i, j, k + 1) < 0.0f))
                    quad(cellId(i - 1, j - 1, k), cellId(i, j - 1, k), cellId(i, j, k), cellId(i - 1, j, k), !in0);
            }

    // Colour: where the vertex came from on A (u) and on B (T(u)), blended
    // the same way as the distances
    if (outColors && HasColor()) {
        outColors->resize(outVerts.size());
        ParallelFor((int)outVerts.size(), [&](int v) {
            glm::vec3 u = LtInv * (outVerts[v] - shift);
            (*outColors)[v] = (1.0f - t) * colorGridA.Sample(u) + t * colorGridB.Sample(A * u + b);
        });
    }

    lastExtractMs = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
}
