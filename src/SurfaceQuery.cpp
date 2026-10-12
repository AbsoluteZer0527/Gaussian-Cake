#include "SurfaceQuery.h"

#include <algorithm>
#include <cmath>

glm::vec3 ClosestPointOnTriangle(const glm::vec3& p, const glm::vec3& a,
                                 const glm::vec3& b, const glm::vec3& c) {
    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 ap = p - a;

    float d1 = glm::dot(ab, ap);
    float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;                   // vertex A region

    glm::vec3 bp = p - b;
    float d3 = glm::dot(ab, bp);
    float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;                     // vertex B region

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {             // edge AB
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    glm::vec3 cp = p - c;
    float d5 = glm::dot(ab, cp);
    float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;                     // vertex C region

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {             // edge AC
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) { // edge BC
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    // Interior projection via barycentric coordinates.
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + v * ab + w * ac;
}

// ---- SurfaceBVH ----

static const int kLeafSize = 4;

void SurfaceBVH::Build(const std::vector<Triangle>& input) {
    tris = input;
    nodes.clear();
    normals.resize(tris.size());
    order.resize(tris.size());
    std::vector<glm::vec3> centroids(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        const Triangle& t = tris[i];
        glm::vec3 n = glm::cross(t.v1 - t.v0, t.v2 - t.v0);
        float len = glm::length(n);
        normals[i] = (len > 1e-20f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
        centroids[i] = (t.v0 + t.v1 + t.v2) / 3.0f;
        order[i] = (int)i;
    }
    if (tris.empty()) return;
    nodes.reserve(2 * tris.size() / kLeafSize + 2);
    BuildNode(0, (int)tris.size(), centroids);
}

int SurfaceBVH::BuildNode(int first, int count, const std::vector<glm::vec3>& centroids) {
    int id = (int)nodes.size();
    nodes.push_back(Node());

    Node n;
    n.lo = glm::vec3(1e30f);
    n.hi = glm::vec3(-1e30f);
    glm::vec3 cLo(1e30f), cHi(-1e30f);
    glm::vec3 weighted(0.0f);
    float area = 0.0f;
    n.areaNormal = glm::vec3(0.0f);
    for (int i = first; i < first + count; ++i) {
        const Triangle& t = tris[order[i]];
        for (const glm::vec3& v : {t.v0, t.v1, t.v2}) {
            n.lo = glm::min(n.lo, v);
            n.hi = glm::max(n.hi, v);
        }
        cLo = glm::min(cLo, centroids[order[i]]);
        cHi = glm::max(cHi, centroids[order[i]]);
        glm::vec3 an = 0.5f * glm::cross(t.v1 - t.v0, t.v2 - t.v0);
        float a = glm::length(an);
        n.areaNormal += an;
        weighted += a * centroids[order[i]];
        area += a;
    }
    n.center = (area > 1e-20f) ? weighted / area : 0.5f * (n.lo + n.hi);
    n.radius = glm::length(glm::max(n.hi - n.center, n.center - n.lo));

    if (count <= kLeafSize) {
        n.first = first;
        n.count = count;
        nodes[id] = n;
        return id;
    }

    // split at the median centroid along the longest axis
    glm::vec3 ext = cHi - cLo;
    int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
    int mid = first + count / 2;
    std::nth_element(order.begin() + first, order.begin() + mid, order.begin() + first + count,
                     [&](int a, int b) { return centroids[a][axis] < centroids[b][axis]; });

    n.left  = BuildNode(first, mid - first, centroids);
    n.right = BuildNode(mid, first + count - mid, centroids);
    nodes[id] = n;
    return id;
}

static float BoxDist2(const glm::vec3& p, const glm::vec3& lo, const glm::vec3& hi) {
    glm::vec3 d = glm::max(glm::vec3(0.0f), glm::max(lo - p, p - hi));
    return glm::dot(d, d);
}

SurfaceBVH::Hit SurfaceBVH::Closest(const glm::vec3& p) const {
    Hit best;
    if (nodes.empty()) return best;

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const Node& n = nodes[stack[--sp]];
        if (BoxDist2(p, n.lo, n.hi) >= best.dist2) continue;

        if (n.left < 0) {
            for (int i = n.first; i < n.first + n.count; ++i) {
                const Triangle& t = tris[order[i]];
                glm::vec3 q = ClosestPointOnTriangle(p, t.v0, t.v1, t.v2);
                float d = glm::dot(q - p, q - p);
                if (d < best.dist2) { best.dist2 = d; best.point = q; best.tri = order[i]; }
            }
            continue;
        }
        // visit the nearer child first (pushed last)
        float dl = BoxDist2(p, nodes[n.left].lo, nodes[n.left].hi);
        float dr = BoxDist2(p, nodes[n.right].lo, nodes[n.right].hi);
        if (dl < dr) { stack[sp++] = n.right; stack[sp++] = n.left; }
        else         { stack[sp++] = n.left;  stack[sp++] = n.right; }
    }
    return best;
}

// Signed solid angle of triangle (a, b, c) seen from p (Van Oosterom & Strackee).
static float SolidAngle(const glm::vec3& p, const Triangle& t) {
    glm::vec3 a = t.v0 - p, b = t.v1 - p, c = t.v2 - p;
    float la = glm::length(a), lb = glm::length(b), lc = glm::length(c);
    float det = glm::dot(a, glm::cross(b, c));
    float den = la * lb * lc + glm::dot(a, b) * lc + glm::dot(b, c) * la + glm::dot(c, a) * lb;
    return 2.0f * std::atan2(det, den);
}

float SurfaceBVH::WindingNumber(const glm::vec3& p) const {
    if (nodes.empty()) return 0.0f;
    const float kPi4 = 4.0f * 3.14159265358979f;
    const float beta = 2.0f;   // accuracy: use the dipole once d > beta * radius

    float w = 0.0f;
    int stack[128];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const Node& n = nodes[stack[--sp]];
        glm::vec3 r = n.center - p;
        float d = glm::length(r);
        if (d > beta * n.radius) {
            w += glm::dot(r, n.areaNormal) / (kPi4 * d * d * d);
            continue;
        }
        if (n.left < 0) {
            for (int i = n.first; i < n.first + n.count; ++i)
                w += SolidAngle(p, tris[order[i]]) / kPi4;
            continue;
        }
        stack[sp++] = n.left;
        stack[sp++] = n.right;
    }
    return w;
}

// ---- PointGrid ----

void PointGrid::Build(const std::vector<glm::vec3>& points, float cellSize) {
    pts = points;
    lo = glm::vec3(1e30f);
    glm::vec3 hi(-1e30f);
    for (const auto& p : pts) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
    if (pts.empty()) { lo = hi = glm::vec3(0.0f); }

    glm::vec3 ext = hi - lo;
    cell = std::max(std::max(1e-4f, cellSize), std::max(ext.x, std::max(ext.y, ext.z)) / 128.0f);
    nx = (int)(ext.x / cell) + 1;
    ny = (int)(ext.y / cell) + 1;
    nz = (int)(ext.z / cell) + 1;

    std::vector<int> count(nx * ny * nz + 1, 0);
    std::vector<int> cellOf(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        glm::ivec3 c = Cell(pts[i]);
        cellOf[i] = (c.z * ny + c.y) * nx + c.x;
        count[cellOf[i] + 1]++;
    }
    for (size_t c = 1; c < count.size(); ++c) count[c] += count[c - 1];
    cellStart = count;
    cellItems.resize(pts.size());
    std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
    for (size_t i = 0; i < pts.size(); ++i) cellItems[fill[cellOf[i]]++] = (int)i;
}

glm::ivec3 PointGrid::Cell(const glm::vec3& p) const {
    glm::ivec3 c = glm::ivec3(glm::floor((p - lo) / cell));
    return glm::clamp(c, glm::ivec3(0), glm::ivec3(nx - 1, ny - 1, nz - 1));
}

// Searches shells of cells outward and stops once the next shell can't
// contain anything closer.
int PointGrid::Nearest(const glm::vec3& q) const {
    if (pts.empty()) return -1;
    glm::ivec3 c0 = Cell(q);
    glm::vec3 hiBox = lo + cell * glm::vec3((float)nx, (float)ny, (float)nz);
    float outside = glm::length(glm::max(glm::vec3(0.0f), glm::max(lo - q, q - hiBox)));

    int best = -1;
    float bestD = 1e30f;
    int maxR = std::max(nx, std::max(ny, nz));
    for (int r = 0; r <= maxR; ++r) {
        for (int z = c0.z - r; z <= c0.z + r; ++z) {
            if (z < 0 || z >= nz) continue;
            for (int y = c0.y - r; y <= c0.y + r; ++y) {
                if (y < 0 || y >= ny) continue;
                for (int x = c0.x - r; x <= c0.x + r; ++x) {
                    if (x < 0 || x >= nx) continue;
                    int ring = std::max(std::abs(x - c0.x), std::max(std::abs(y - c0.y), std::abs(z - c0.z)));
                    if (ring != r) continue;
                    int c = (z * ny + y) * nx + x;
                    for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                        glm::vec3 d = pts[cellItems[k]] - q;
                        float dd = glm::dot(d, d);
                        if (dd < bestD) { bestD = dd; best = cellItems[k]; }
                    }
                }
            }
        }
        float reach2 = outside * outside + (r * cell) * (r * cell);
        if (best >= 0 && bestD <= reach2) break;
    }
    return best;
}
