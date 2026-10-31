#pragma once
#include "core.h"
#include "Texture.h"
#include <memory>
#include <vector>

// Colours here are display values (gamma-encoded, like the pixels of a
// texture and like Splat::color), so they can go straight into a splat.

struct Triangle {
    glm::vec3 v0, v1, v2;

    // Appearance. Meshes without it keep these defaults, so {a, b, c} still
    // makes a plain triangle.
    glm::vec2 uv0{0.0f}, uv1{0.0f}, uv2{0.0f};   // texture coords, v = 0 at the image's top row
    glm::vec3 c0{1.0f}, c1{1.0f}, c2{1.0f};      // vertex colours
    int material = -1;                           // index into MeshData::materials, -1 = none
};

struct Material {
    glm::vec3 baseColor{1.0f};
    // shared_ptr: several materials (and copies of the mesh) can share one
    // image, and it's freed when the last one goes away
    std::shared_ptr<Texture> texture;            // null = no texture
};

// A triangle mesh plus what it looks like.
struct MeshData {
    std::vector<Triangle> tris;
    std::vector<Material> materials;
    bool hasVertexColors = false;

    // Has any colour beyond the default grey?
    bool HasAppearance() const { return hasVertexColors || !materials.empty(); }

    // Colour of triangle `tri` at barycentric coordinates `bary`
    // (bary.x for v0, bary.y for v1, bary.z for v2; they sum to 1).
    glm::vec3 Albedo(int tri, const glm::vec3& bary) const;
};
