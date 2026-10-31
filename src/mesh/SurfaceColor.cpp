#include "SurfaceColor.h"
#include "Parallel.h"

void SurfaceColor::Build(const MeshData& source, const glm::vec3& fallbackColor) {
    fallback = fallbackColor;
    mesh = MeshData();
    bvh = SurfaceBVH();
    if (!source.HasAppearance()) return;
    mesh = source;
    bvh.Build(mesh.tris);
}

glm::vec3 SurfaceColor::At(const glm::vec3& p) const {
    if (!HasColor()) return fallback;
    SurfaceBVH::Hit hit = bvh.Closest(p);
    if (hit.tri < 0) return fallback;
    const Triangle& t = mesh.tris[hit.tri];
    return mesh.Albedo(hit.tri, Barycentric(hit.point, t.v0, t.v1, t.v2));
}

std::vector<glm::vec3> SurfaceColor::At(const std::vector<glm::vec3>& points) const {
    std::vector<glm::vec3> out(points.size(), fallback);
    if (HasColor())
        ParallelFor((int)points.size(), [&](int i) { out[i] = At(points[i]); });
    return out;
}
