#pragma once
#include "core.h"
#include "OBJLoader.h"
#include "Splat.h"
#include <vector>

class SplatGenerator {
public:
    // Sample N surface-aligned disk splats from a triangle mesh (area-weighted).
    // Each splat lies flat in its triangle's plane; tangent std-dev is
    // sizeMul * sqrt(totalArea / N) so neighboring splats overlap.
    static std::vector<Splat> FromTriangles(const std::vector<Triangle>& tris,
                                            int N,
                                            float sizeMul,
                                            unsigned int seed = 42);

    // Same, coloured by the mesh's materials, textures and vertex colours
    // (grey, as above, when it has none).
    static std::vector<Splat> FromMesh(const MeshData& mesh,
                                       int N,
                                       float sizeMul,
                                       unsigned int seed = 42);

    // Typical tangent std-dev for N splats on this surface (before sizeMul).
    static float BaseSigma(const std::vector<Triangle>& tris, int N);

    // Rotation whose local z axis is the triangle normal.
    static glm::quat TriangleFrame(const Triangle& t);

    // Unit normal of a triangle (falls back to +Y for degenerate triangles).
    static glm::vec3 TriangleNormal(const Triangle& t);

    // Bake the same Lambert shading as shaders/shader.frag into a color.
    // Uses |n.L| so both sides of a disk are lit.
    static glm::vec3 ShadeNormal(const glm::vec3& n);

    // The same shading on a surface of colour `albedo` (a display value).
    static glm::vec3 ShadeNormal(const glm::vec3& n, const glm::vec3& albedo);

    static glm::vec3 baseColor;
};
