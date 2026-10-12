#pragma once
#include "core.h"
#include "OBJLoader.h"
#include <vector>

// Closest point on triangle (a, b, c) to p.
// Ericson, "Real-Time Collision Detection" §5.1.5 — Voronoi-region test.
glm::vec3 ClosestPointOnTriangle(const glm::vec3& p, const glm::vec3& a,
                                 const glm::vec3& b, const glm::vec3& c);

// AABB tree over a triangle soup: exact closest-point queries and the
// generalized winding number (Jacobson et al. 2013) for inside/outside tests
// that still work on open or multi-part meshes.
class SurfaceBVH {
public:
    struct Hit {
        glm::vec3 point;
        int       tri = -1;
        float     dist2 = 1e30f;
    };

    void Build(const std::vector<Triangle>& tris);

    Hit Closest(const glm::vec3& p) const;

    // Sum of solid angles / 4pi: ~1 inside a closed outward-facing mesh, ~0
    // outside. Far-away clusters use a dipole approximation (Barill et al. 2018).
    float WindingNumber(const glm::vec3& p) const;

    // Unit face normal of triangle t.
    glm::vec3 FaceNormal(int t) const { return normals[t]; }

    bool Empty() const { return tris.empty(); }

private:
    struct Node {
        glm::vec3 lo, hi;
        glm::vec3 center;       // area-weighted centroid
        glm::vec3 areaNormal;   // sum of area * normal (dipole moment)
        float     radius;       // bounds every point of the node around center
        int left = -1, right = -1;
        int first = 0, count = 0;   // leaf range in order[]
    };

    std::vector<Triangle>  tris;
    std::vector<glm::vec3> normals;
    std::vector<int>       order;
    std::vector<Node>      nodes;

    int BuildNode(int first, int count, const std::vector<glm::vec3>& centroids);
};

// Uniform grid over points for nearest-point queries.
class PointGrid {
public:
    void Build(const std::vector<glm::vec3>& points, float cellSize);
    int  Nearest(const glm::vec3& q) const;   // index into the points, -1 if empty

private:
    std::vector<glm::vec3> pts;
    std::vector<int> cellStart, cellItems;   // CSR: cellStart[c]..cellStart[c+1]
    glm::vec3 lo = glm::vec3(0.0f);
    float cell = 1.0f;
    int nx = 1, ny = 1, nz = 1;

    glm::ivec3 Cell(const glm::vec3& p) const;
};
