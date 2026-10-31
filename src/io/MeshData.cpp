#include "MeshData.h"

// albedo = vertex colour x material base colour x texture colour, each
// blended across the triangle with the point's barycentric weights
glm::vec3 MeshData::Albedo(int tri, const glm::vec3& bary) const {
    const Triangle& t = tris[tri];
    glm::vec2 uv = bary.x * t.uv0 + bary.y * t.uv1 + bary.z * t.uv2;
    glm::vec3 color = bary.x * t.c0 + bary.y * t.c1 + bary.z * t.c2;

    if (t.material >= 0 && t.material < (int)materials.size()) {
        const Material& m = materials[t.material];
        color *= m.baseColor;
        if (m.texture) color *= m.texture->Sample(uv);
    }
    return color;
}
