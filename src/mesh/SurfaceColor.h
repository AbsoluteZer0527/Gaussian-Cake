#pragma once
#include "core.h"
#include "MeshData.h"
#include "SurfaceQuery.h"
#include <vector>

// The colour a mesh gives to any point in space: the albedo at the closest
// point of its surface. That's how surfaces built from the mesh (a template
// fitted onto it, the SDF morph's in-between surfaces) get its colours.
// A mesh without colours answers `fallback` everywhere.
class SurfaceColor {
public:
    void Build(const MeshData& mesh, const glm::vec3& fallback);

    bool HasColor() const { return !bvh.Empty(); }
    glm::vec3 At(const glm::vec3& p) const;
    std::vector<glm::vec3> At(const std::vector<glm::vec3>& points) const;   // in parallel

private:
    MeshData   mesh;
    SurfaceBVH bvh;
    glm::vec3  fallback{1.0f};
};
