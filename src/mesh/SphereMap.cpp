#include "SphereMap.h"
#include "Laplacian.h"
#include "Parallel.h"
#include "SurfaceQuery.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <glm/gtx/quaternion.hpp>

namespace SphereMap {

static glm::vec3 AreaCentroid(const IndexedMesh& m, const std::vector<glm::vec3>& x, double* totalArea = nullptr) {
    glm::dvec3 c(0.0);
    double total = 0.0;
    for (int f = 0; f < m.FaceCount(); ++f) {
        glm::vec3 a = x[m.indices[3 * f]], b = x[m.indices[3 * f + 1]], d = x[m.indices[3 * f + 2]];
        double area = 0.5 * glm::length(glm::cross(b - a, d - a));
        c += area * glm::dvec3((a + b + d) / 3.0f);
        total += area;
    }
    if (totalArea) *totalArea = total;
    return total > 0.0 ? glm::vec3(c / total) : glm::vec3(0.0f);
}

int CountFlipped(const IndexedMesh& m, const std::vector<glm::vec3>& s) {
    int flipped = 0;
    for (int f = 0; f < m.FaceCount(); ++f) {
        glm::vec3 a = s[m.indices[3 * f]], b = s[m.indices[3 * f + 1]], c = s[m.indices[3 * f + 2]];
        if (glm::dot(glm::cross(b - a, c - a), a + b + c) <= 0.0f) ++flipped;
    }
    return flipped;
}

// Spherical Tutte relaxation of the inside-out triangles: their vertices
// (and a growing ring around them) move to the normalized mean of their
// neighbours, until none are left.
static int Untangle(const IndexedMesh& m, const std::vector<std::vector<int>>& rings,
                    std::vector<glm::vec3>& s, int maxRounds) {
    const int n = (int)s.size();
    int grow = 1;
    int best = CountFlipped(m, s), stuck = 0;
    for (int r = 0; r < maxRounds && best > 0; ++r) {
        std::vector<char> move(n, 0);
        for (int f = 0; f < m.FaceCount(); ++f) {
            glm::vec3 a = s[m.indices[3 * f]], b = s[m.indices[3 * f + 1]], c = s[m.indices[3 * f + 2]];
            if (glm::dot(glm::cross(b - a, c - a), a + b + c) <= 0.0f)
                for (int k = 0; k < 3; ++k) move[m.indices[3 * f + k]] = 1;
        }
        for (int g = 0; g < grow; ++g) {
            std::vector<char> wider = move;
            for (int i = 0; i < n; ++i)
                if (move[i]) for (int j : rings[i]) wider[j] = 1;
            move.swap(wider);
        }
        for (int sweep = 0; sweep < 5; ++sweep) {
            std::vector<glm::vec3> next = s;
            ParallelFor(n, [&](int i) {
                if (!move[i] || rings[i].empty()) return;
                glm::vec3 c(0.0f);
                for (int j : rings[i]) c += s[j];
                float len = glm::length(c);
                if (len > 1e-12f) next[i] = c / len;
            });
            s.swap(next);
        }
        int now = CountFlipped(m, s);
        if (now < best) { best = now; stuck = 0; }
        else if (++stuck >= 2) { grow = std::min(grow * 2, 64); stuck = 0; }
    }
    return best;
}

// Moebius map of the unit sphere that sends c (inside the ball) to the centre.
static glm::vec3 Moebius(const glm::vec3& x, const glm::vec3& c) {
    float c2 = glm::dot(c, c);
    glm::vec3 d = x - c;
    float d2 = glm::dot(d, d);
    glm::vec3 y = ((1.0f - c2) * d - d2 * c) / std::max(1.0f - 2.0f * glm::dot(c, x) + c2 * glm::dot(x, x), 1e-12f);
    return glm::normalize(y);
}

bool Compute(const IndexedMesh& m, std::vector<glm::vec3>& s, SphereStats* stats) {
    auto t0 = std::chrono::high_resolution_clock::now();
    SphereStats st;
    const int n = (int)m.verts.size();
    s.clear();
    TopologyInfo info = MeshTopology::Analyze(m);
    if (n < 4 || info.parts != 1 || info.genus != 0 || info.boundaryLoops != 0 || info.nonManifoldEdges != 0) {
        if (stats) *stats = st;
        return false;
    }

    // Conformalized mean-curvature flow: the stiffness stays the original
    // mesh's (so it can't pinch), only the vertex areas follow the shape.
    // Each step: (M + dt L0) x' = M x, then re-centre and scale to unit area.
    std::vector<glm::vec3> x = m.verts;
    {
        double area;
        glm::vec3 c = AreaCentroid(m, x, &area);
        float k = (float)(1.0 / std::sqrt(area));
        for (glm::vec3& p : x) p = (p - c) * k;
    }
    SpMat L = CotanMatrix(m, x);
    Eigen::SimplicialLDLT<SpMat> solver;
    bool analyzed = false;
    const double dt = 0.05;
    for (int step = 0; step < 40; ++step) {
        std::vector<double> mass = VertexAreas(m, x);
        SpMat A = dt * L;
        Eigen::MatrixXd rhs(n, 3);
        for (int i = 0; i < n; ++i) {
            A.coeffRef(i, i) += std::max(mass[i], 1e-12);
            rhs.row(i) = mass[i] * ToE(x[i]).transpose();
        }
        if (!analyzed) { solver.analyzePattern(A); analyzed = true; }
        solver.factorize(A);
        if (solver.info() != Eigen::Success) break;
        Eigen::MatrixXd sol = solver.solve(rhs);
        for (int i = 0; i < n; ++i) x[i] = glm::vec3((float)sol(i, 0), (float)sol(i, 1), (float)sol(i, 2));
        double area;
        glm::vec3 c = AreaCentroid(m, x, &area);
        if (!(area > 1e-20)) break;
        float k = (float)(1.0 / std::sqrt(area));
        float lo = 1e30f, hi = 0.0f;
        double sum = 0.0;
        for (glm::vec3& p : x) {
            p = (p - c) * k;
            float r = glm::length(p);
            lo = std::min(lo, r);
            hi = std::max(hi, r);
            sum += r;
        }
        st.flowSteps = step + 1;
        st.roundness = (float)((hi - lo) / (sum / n));
        if (st.roundness < 0.02f) break;
    }

    // onto the sphere
    s.resize(n);
    for (int i = 0; i < n; ++i) {
        float r = glm::length(x[i]);
        s[i] = r > 1e-12f ? x[i] / r : glm::vec3(0, 0, 1);
    }
    auto rings = MeshTopology::OneRings(m);
    st.flippedAfterFlow = CountFlipped(m, s);
    Untangle(m, rings, s, 200);

    // Spread the area: move the centre of the original surface's area (as it
    // sits on the sphere) to the sphere's centre, a damped step at a time
    {
        std::vector<double> faceArea(m.FaceCount());
        double total = 0.0;
        for (int f = 0; f < m.FaceCount(); ++f) {
            const glm::vec3& a = m.verts[m.indices[3 * f]];
            faceArea[f] = 0.5 * glm::length(glm::cross(m.verts[m.indices[3 * f + 1]] - a, m.verts[m.indices[3 * f + 2]] - a));
            total += faceArea[f];
        }
        for (int it = 0; it < 30 && total > 0.0; ++it) {
            glm::dvec3 g(0.0);
            for (int f = 0; f < m.FaceCount(); ++f) {
                glm::vec3 c = s[m.indices[3 * f]] + s[m.indices[3 * f + 1]] + s[m.indices[3 * f + 2]];
                float len = glm::length(c);
                if (len > 1e-12f) g += (faceArea[f] / total) * glm::dvec3(c / len);
            }
            glm::vec3 gc = glm::vec3(g);
            if (glm::length(gc) < 1e-3f) break;
            glm::vec3 c = 0.5f * gc;   // damped
            std::vector<glm::vec3> next(n);
            for (int i = 0; i < n; ++i) next[i] = Moebius(s[i], c);
            // keep the step only if it doesn't turn triangles inside out
            if (CountFlipped(m, next) > CountFlipped(m, s)) break;
            s.swap(next);
        }
    }
    st.flipped = Untangle(m, rings, s, 50);
    st.ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    if (stats) *stats = st;
    return st.flipped == 0;
}

// ---- Moebius maps through the complex plane (stereographic projection) ----
using C = std::complex<double>;
static const double kInf = 1e18;

// From the sphere to the plane, projecting from the north pole (0, 0, 1);
// the pole itself goes to (practically) infinity.
static C ToPlane(const glm::vec3& p) {
    double d = 1.0 - p.z;
    if (d < 1e-12) return C(kInf, 0.0);
    return C(p.x / d, p.y / d);
}
static glm::vec3 ToSphere(C z) {
    double r2 = std::norm(z);
    if (!std::isfinite(r2) || r2 > 1e30) return glm::vec3(0, 0, 1);
    return glm::vec3((float)(2.0 * z.real() / (r2 + 1.0)), (float)(2.0 * z.imag() / (r2 + 1.0)),
                     (float)((r2 - 1.0) / (r2 + 1.0)));
}
struct M2 { C a, b, c, d; };   // z -> (a z + b) / (c z + d)
static M2 Mul(const M2& p, const M2& q) {
    return { p.a * q.a + p.b * q.c, p.a * q.b + p.b * q.d, p.c * q.a + p.d * q.c, p.c * q.b + p.d * q.d };
}
static M2 Inv(const M2& m) { return { m.d, -m.b, -m.c, m.a }; }
// The map sending z1, z2, z3 to 0, 1, infinity.
static M2 ToStandard(C z1, C z2, C z3) {
    return { z2 - z3, -z1 * (z2 - z3), z2 - z1, -z3 * (z2 - z1) };
}

void MoebiusFit(const std::vector<glm::vec3>& from, const std::vector<glm::vec3>& to,
                std::vector<glm::vec3>& points) {
    size_t k = std::min<size_t>(3, std::min(from.size(), to.size()));
    if (k == 0) return;
    std::vector<glm::vec3> a(from.begin(), from.begin() + k), b(to.begin(), to.begin() + k);
    // fewer than 3 pairs: hold extra points still, as far from the pins as possible
    if (k == 1) {
        glm::vec3 axis = glm::normalize(a[0] + b[0]);
        if (!std::isfinite(axis.x) || glm::length(a[0] + b[0]) < 1e-3f) axis = a[0];
        glm::vec3 side = glm::normalize(glm::cross(axis, std::abs(axis.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0)));
        a.push_back(-axis); b.push_back(-axis);
        a.push_back(side);  b.push_back(side);
    } else if (k == 2) {
        glm::vec3 far = -(a[0] + a[1] + b[0] + b[1]);
        far = glm::length(far) > 1e-3f ? glm::normalize(far) : glm::normalize(glm::cross(a[0], a[1]));
        a.push_back(far); b.push_back(far);
    }
    // rotate so no key point sits at the projection pole
    glm::vec3 avoid(0, 0, 1);
    for (int tries = 0; tries < 8; ++tries) {
        bool clear = true;
        for (int i = 0; i < 3; ++i)
            if (glm::dot(a[i], avoid) > 0.99f || glm::dot(b[i], avoid) > 0.99f) clear = false;
        if (clear) break;
        avoid = glm::normalize(glm::vec3(std::sin(1.3f * (tries + 1)), std::cos(2.1f * (tries + 1)), 0.3f));
    }
    // a rotation taking `avoid` to the pole
    glm::quat q = glm::rotation(avoid, glm::vec3(0, 0, 1));
    glm::quat qi = glm::inverse(q);
    C z[3], w[3];
    for (int i = 0; i < 3; ++i) { z[i] = ToPlane(q * a[i]); w[i] = ToPlane(q * b[i]); }
    M2 f = Mul(Inv(ToStandard(w[0], w[1], w[2])), ToStandard(z[0], z[1], z[2]));
    for (glm::vec3& p : points) {
        C x = ToPlane(q * p);
        C y;
        if (std::abs(x) > 1e17) y = f.a / f.c;    // the pole goes to a / c
        else {
            C den = f.c * x + f.d;
            y = std::abs(den) < 1e-300 ? C(kInf, 0.0) : (f.a * x + f.b) / den;
        }
        p = glm::normalize(qi * ToSphere(y));
    }
}

glm::mat3 Kabsch(const std::vector<std::pair<glm::vec3, glm::vec3>>& pairs, const std::vector<float>& w) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < pairs.size(); ++i)
        H += (double)w[i] * ToE(pairs[i].first) * ToE(pairs[i].second).transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0.0) {
        V.col(2) *= -1.0;
        R = V * U.transpose();
    }
    glm::mat3 out;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[c][r] = (float)R(r, c);
    return out;
}

static std::vector<Triangle> Tris(const IndexedMesh& m, const std::vector<glm::vec3>& pos) {
    std::vector<Triangle> t(m.FaceCount());
    for (int f = 0; f < m.FaceCount(); ++f)
        t[f] = { pos[m.indices[3 * f]], pos[m.indices[3 * f + 1]], pos[m.indices[3 * f + 2]] };
    return t;
}

glm::vec3 SpherePointOf(const IndexedMesh& target, const std::vector<glm::vec3>& targetSphere,
                        const glm::vec3& onSurface) {
    // (one-off queries: a fresh BVH each call would be slow, so callers batch;
    // this is used only for pins)
    SurfaceBVH bvh;
    bvh.Build(Tris(target, target.verts));
    SurfaceBVH::Hit h = bvh.Closest(onSurface);
    if (h.tri < 0) return glm::vec3(0, 0, 1);
    unsigned int a = target.indices[3 * h.tri], b = target.indices[3 * h.tri + 1], c = target.indices[3 * h.tri + 2];
    glm::vec3 w = Barycentric(h.point, target.verts[a], target.verts[b], target.verts[c]);
    return glm::normalize(w.x * targetSphere[a] + w.y * targetSphere[b] + w.z * targetSphere[c]);
}

std::vector<glm::vec3> Transfer(const std::vector<glm::vec3>& points, const IndexedMesh& target,
                                const std::vector<glm::vec3>& targetSphere) {
    SurfaceBVH bvh;
    bvh.Build(Tris(target, targetSphere));
    std::vector<glm::vec3> out(points.size());
    ParallelFor((int)points.size(), [&](int i) {
        SurfaceBVH::Hit h = bvh.Closest(points[i]);
        if (h.tri < 0) { out[i] = points[i]; return; }
        unsigned int a = target.indices[3 * h.tri], b = target.indices[3 * h.tri + 1], c = target.indices[3 * h.tri + 2];
        glm::vec3 w = Barycentric(h.point, targetSphere[a], targetSphere[b], targetSphere[c]);
        out[i] = w.x * target.verts[a] + w.y * target.verts[b] + w.z * target.verts[c];
    });
    return out;
}

}  // namespace SphereMap
