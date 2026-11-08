#include "SDFMorph.h"
#include "SurfaceQuery.h"
#include "Parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

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

glm::vec3 WarpGrid::Sample(const glm::vec3& u, glm::mat3* outJ) const {
    glm::vec3 g = (u - lo) / h;
    glm::vec3 gc = glm::clamp(g, glm::vec3(0.0f), glm::vec3((float)(n - 1) - 1e-4f));
    int i = (int)gc.x, j = (int)gc.y, k = (int)gc.z;
    glm::vec3 f = gc - glm::vec3((float)i, (float)j, (float)k);
    auto at = [&](int a, int b, int c) { return ((size_t)c * n + b) * n + a; };
    float w[8];
    size_t id[8];
    int q = 0;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx, ++q) {
                w[q] = (dx ? f.x : 1.0f - f.x) * (dy ? f.y : 1.0f - f.y) * (dz ? f.z : 1.0f - f.z);
                id[q] = at(i + dx, j + dy, k + dz);
            }
    glm::vec3 t(0.0f);
    glm::mat3 Jm(0.0f);
    for (int c = 0; c < 8; ++c) {
        t  += w[c] * T[id[c]];
        Jm += w[c] * J[id[c]];
    }
    // past the grid: continue along the map's own slope
    glm::vec3 off = h * (g - gc);
    t += Jm * off;
    if (outJ) *outJ = Jm;
    return t;
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

// ---- The mixture warp: grid helpers (n^3 points, x fastest) ----

static size_t GridAt(int n, const int p[3]) { return ((size_t)p[2] * n + p[1]) * n + p[0]; }

static void GridSplit(int n, int idx, int p[3]) {
    p[0] = idx % n;
    p[1] = (idx / n) % n;
    p[2] = idx / (n * n);
}

// One pass along each axis in turn: each point becomes `reduce` over its 2r+1
// neighbours on that axis (clamped at the border).
template <typename V, typename Reduce>
static void SeparableFilter(std::vector<V>& f, int n, int r, Reduce reduce) {
    std::vector<V> out(f.size());
    for (int axis = 0; axis < 3; ++axis) {
        ParallelFor((int)f.size(), [&](int idx) {
            int p[3];
            GridSplit(n, idx, p);
            auto neighbour = [&](int d) {
                int q[3] = { p[0], p[1], p[2] };
                q[axis] = std::clamp(p[axis] + d, 0, n - 1);
                return f[GridAt(n, q)];
            };
            out[idx] = reduce(neighbour);
        });
        f.swap(out);
    }
}

template <typename V>
static void BoxBlur(std::vector<V>& f, int n, int r) {
    SeparableFilter(f, n, r, [r](auto neighbour) {
        V sum = neighbour(-r);
        for (int d = -r + 1; d <= r; ++d) sum += neighbour(d);
        return sum / (float)(2 * r + 1);
    });
}

static void MinFilter(std::vector<float>& f, int n, int r) {
    SeparableFilter(f, n, r, [r](auto neighbour) {
        float m = neighbour(-r);
        for (int d = -r + 1; d <= r; ++d) m = std::min(m, neighbour(d));
        return m;
    });
}

// Jacobian of a vec3 grid at one point, by central differences (one-sided at
// the border). Column `axis` = d/du_axis.
static glm::mat3 GridJacobian(const std::vector<glm::vec3>& v, int n, float h, int idx) {
    int p[3];
    GridSplit(n, idx, p);
    glm::mat3 J;
    for (int axis = 0; axis < 3; ++axis) {
        int a[3] = { p[0], p[1], p[2] }, c[3] = { p[0], p[1], p[2] };
        a[axis] = std::max(p[axis] - 1, 0);
        c[axis] = std::min(p[axis] + 1, n - 1);
        J[axis] = (v[GridAt(n, c)] - v[GridAt(n, a)]) / ((c[axis] - a[axis]) * h);
    }
    return J;
}

// The mixture map, made fold-free. As it is, it can fold: near a border
// between parts the blended maps differ a lot, and then some
// M_t = (1 - t) Id + t T stops being one-to-one (holes and floaters
// mid-morph). So the morph uses
//   T = affine map + lambda(u) * smooth(mixture map - affine map)
// with lambda in [0, 1] as large as it can be while no M_t folds anywhere.
// (lambda = 0 is the affine map, which never folds: A is SPD.)
// Returns false if that fails; strength = the mean lambda.
static bool BuildWarp(const MixtureOTMap& mixture, const glm::mat3& A, const glm::vec3& b,
                      int n, const glm::vec3& lo, float h, WarpGrid& warp, float& strength) {
    const size_t count = (size_t)n * n * n;
    auto point = [&](int idx) {
        int p[3];
        GridSplit(n, idx, p);
        return lo + h * glm::vec3((float)p[0], (float)p[1], (float)p[2]);
    };

    // 1. The residual R = mixture map - affine map, smoothed (three box blurs
    //    per axis, close to a Gaussian with sigma ~ 2 cells), and its Jacobian
    std::vector<glm::vec3> resid(count);
    ParallelFor((int)count, [&](int idx) {
        glm::vec3 u = point(idx), Tu;
        Eigen::Matrix3d unused;
        OptimalTransport::ApplyMixture(mixture, u, Tu, unused);
        resid[idx] = Tu - (A * u + b);
    });
    for (int pass = 0; pass < 3; ++pass) BoxBlur(resid, n, 2);
    std::vector<glm::mat3> residJ(count);
    ParallelFor((int)count, [&](int idx) { residJ[idx] = GridJacobian(resid, n, h, idx); });

    // M_t' = (1 - t) I + t J must keep a positive determinant for every t
    // (a cubic in t; checked at 16 points, with a small margin)
    auto folds = [](const glm::mat3& J) {
        for (int s = 1; s <= 16; ++s) {
            float t = s / 16.0f;
            if (glm::determinant((1.0f - t) * glm::mat3(1.0f) + t * J) <= 0.05f) return true;
        }
        return false;
    };

    // 2. Each grid point's own safe lambda, by bisection
    std::vector<float> lambda(count);
    ParallelFor((int)count, [&](int idx) {
        if (!folds(A + residJ[idx])) { lambda[idx] = 1.0f; return; }
        float lo = 0.0f, hi = 1.0f;
        for (int it = 0; it < 8; ++it) {
            float mid = 0.5f * (lo + hi);
            if (folds(A + mid * residJ[idx])) hi = mid; else lo = mid;
        }
        lambda[idx] = lo;
    });

    // 3. A smooth lambda field: erode (a point is only as strong as its
    //    weakest neighbours), blur, build T and check its real Jacobian
    //    (lambda varies now, so J gains a grad(lambda) x R term). Halve
    //    lambda around every point that still folds, and repeat.
    warp.n = n;
    warp.lo = lo;
    warp.h = h;
    warp.T.resize(count);
    warp.J.resize(count);
    std::vector<char> bad(count, 0);
    for (int round = 0; round < 12; ++round) {
        std::vector<float> lam = lambda;
        MinFilter(lam, n, 2);
        BoxBlur(lam, n, 2);
        BoxBlur(lam, n, 2);
        ParallelFor((int)count, [&](int idx) { warp.T[idx] = A * point(idx) + b + lam[idx] * resid[idx]; });
        ParallelFor((int)count, [&](int idx) {
            warp.J[idx] = GridJacobian(warp.T, n, h, idx);
            bad[idx] = folds(warp.J[idx]) ? 1 : 0;
        });
        strength = (float)(std::accumulate(lam.begin(), lam.end(), 0.0) / count);
        if (std::count(bad.begin(), bad.end(), 1) == 0) return strength > 0.0f;

        ParallelFor((int)count, [&](int idx) {
            int p[3];
            GridSplit(n, idx, p);
            for (int dz = -2; dz <= 2; ++dz)
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        int q[3] = { std::clamp(p[0] + dx, 0, n - 1), std::clamp(p[1] + dy, 0, n - 1),
                                     std::clamp(p[2] + dz, 0, n - 1) };
                        if (bad[GridAt(n, q)]) { lambda[idx] *= 0.5f; return; }
                    }
        });
    }
    return false;
}

void SDFMorph::Build(const std::vector<Triangle>& trisA, const std::vector<Triangle>& trisB,
                     const OTMap& map, int resolution,
                     const SurfaceColor* colorA, const SurfaceColor* colorB,
                     const MixtureOTMap* mixture) {
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

    // The mixture map on the grid. A mixture of one Gaussian is the affine map
    // again, so then there's nothing to store.
    warp = WarpGrid();
    warpStrength = 0.0f;
    if (mixture && mixture->source.K() > 1 &&
        !BuildWarp(*mixture, A, b, resolution, gridA.lo, gridA.h, warp, warpStrength)) {
        warp = WarpGrid();   // couldn't make it fold-free: plain affine morph
        warpStrength = 0.0f;
    }

    built = true;
    buildMs = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t0).count();
}

glm::vec3 SDFMorph::Preimage(const glm::vec3& y, float t, glm::vec3* outTu, glm::mat3* outJ) const {
    // affine answer: exact without a warp, the starting guess with one
    glm::mat3 Lt = (1.0f - t) * glm::mat3(1.0f) + t * A;
    glm::vec3 u = glm::inverse(Lt) * (y - t * b);
    if (warp.Empty()) {
        if (outTu) *outTu = A * u + b;
        if (outJ) *outJ = A;
        return u;
    }
    // Newton on F(u) = (1 - t) u + t T(u) - y, F'(u) = (1 - t) I + t J(u)
    glm::mat3 J;
    glm::vec3 Tu = warp.Sample(u, &J);
    for (int it = 0; it < 10; ++it) {
        glm::vec3 F = (1.0f - t) * u + t * Tu - y;
        if (glm::dot(F, F) < 1e-12f) break;
        glm::mat3 D = (1.0f - t) * glm::mat3(1.0f) + t * J;
        float det = glm::determinant(D);
        if (std::abs(det) < 1e-6f) break;   // a fold: keep the best guess so far
        u -= glm::inverse(D) * F;
        Tu = warp.Sample(u, &J);
    }
    if (outTu) *outTu = Tu;
    if (outJ) *outJ = J;
    return u;
}

float SDFMorph::AlignmentError(const std::vector<glm::vec3>& pointsOnA, bool affineOnly) const {
    if (!built || pointsOnA.empty()) return 0.0f;
    double sum = 0.0;
    for (const glm::vec3& x : pointsOnA) {
        glm::vec3 y = (affineOnly || warp.Empty()) ? A * x + b : warp.Sample(x);
        sum += std::abs(gridB.Sample(y));
    }
    return (float)(sum / pointsOnA.size());
}

void SDFMorph::Field(float t, int stride, int& outN, glm::vec3& outLo, float& outH,
                     std::vector<float>& f) const {
    t = glm::clamp(t, 0.0f, 1.0f);
    stride = std::max(1, stride);
    const int n = (gridA.n - 1) / stride + 1;
    const glm::vec3 lo = gridA.lo;
    const float h = gridA.h * stride;
    outN = n;
    outLo = lo;
    outH = h;

    // M_t(x) = L_t x + t b,  L_t = (1 - t) I + t A  (A is SPD, so L_t is invertible)
    glm::mat3 Lt = (1.0f - t) * glm::mat3(1.0f) + t * A;
    glm::mat3 LtInv = glm::inverse(Lt);
    glm::vec3 shift = t * b;

    f.assign((size_t)n * n * n, 0.0f);
    ParallelFor(n * n * n, [&](int idx) {
        int i = idx % n, j = (idx / n) % n, k = idx / (n * n);
        glm::vec3 y = lo + h * glm::vec3((float)i, (float)j, (float)k);
        glm::vec3 u, Tu;
        if (warp.Empty()) {   // closed form (Preimage would redo LtInv per point)
            u  = LtInv * (y - shift);
            Tu = A * u + b;
        } else {
            u = Preimage(y, t, &Tu);
        }
        float fa = (t < 1.0f) ? gridA.Sample(u) : 0.0f;
        float fb = (t > 0.0f) ? gridB.Sample(Tu) : 0.0f;
        f[idx] = (1.0f - t) * fa + t * fb;
    });
}

void SDFMorph::Extract(float t, std::vector<glm::vec3>& outVerts,
                       std::vector<unsigned int>& outIndices,
                       std::vector<glm::vec3>* outColors, int stride) const {
    outVerts.clear();
    outIndices.clear();
    if (outColors) outColors->clear();
    if (!built) return;
    auto t0 = std::chrono::high_resolution_clock::now();

    t = glm::clamp(t, 0.0f, 1.0f);
    int n;
    glm::vec3 lo;
    float h;
    std::vector<float> f;
    Field(t, stride, n, lo, h, f);
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
            glm::vec3 Tu;
            glm::vec3 u = Preimage(outVerts[v], t, &Tu);
            (*outColors)[v] = (1.0f - t) * colorGridA.Sample(u) + t * colorGridB.Sample(Tu);
        });
    }

    lastExtractMs = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
}
