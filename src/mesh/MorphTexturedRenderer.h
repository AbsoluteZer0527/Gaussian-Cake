#pragma once
#include "core.h"
#include "MeshData.h"
#include "SurfaceColor.h"
#include <vector>

// Draws the shared-topology morph with both meshes' real textures, per pixel,
// the whole way through (instead of colours sampled only at the vertices).
//
// The fit gives every template corner a point on A and a point on B. A colour
// pulled back through that match is just "look it up there" (pullback of a
// 0-form, phi* c = c o phi), so each corner carries A's uv and B's uv, and the
// fragment shader shows mix(texture A at uvA, texture B at uvB, t).
// UVs belong to corners, not to vertices: two triangles meeting at a UV seam
// need different uvs at the same point, so every triangle gets its own 3 corners.

// Where one template corner lands in A's and in B's texture.
struct CornerUV {
    glm::vec2 uvA{0.0f};
    glm::vec2 uvB{0.0f};
    glm::vec3 tintA{1.0f};   // what multiplies texture A there: vertex colour x material colour
    glm::vec3 tintB{1.0f};
    int matA = -1;           // A's textured material there; -1 = use the vertex colour instead
    int matB = -1;
};

// For every corner of the template (3 per triangle, in index order): its uv,
// tint and material on A, found at the closest point of A's surface to
// startPositions, and the same on B from endPositions. A triangle whose
// corners land on different materials or straddle a UV seam (its uvs spread
// far more than its size) gets -1 on that side, so it falls back to vertex
// colours there instead of smearing across the texture.
std::vector<CornerUV> PullBackCorners(const std::vector<glm::vec3>& startPositions,
                                      const std::vector<glm::vec3>& endPositions,
                                      const std::vector<unsigned int>& indices,
                                      const SurfaceColor& colorA, const SurfaceColor& colorB);

class MorphTexturedRenderer {
public:
    ~MorphTexturedRenderer();

    // startPositions / endPositions: the template's vertices at A and at B.
    // startColors / endColors: per-vertex colours, used where a side has no texture.
    // corners: PullBackCorners' result. dataA / dataB: the meshes that own the
    // images; their GL textures come from (and stay in) `textures`.
    void Initialize(const std::vector<glm::vec3>& startPositions,
                    const std::vector<glm::vec3>& endPositions,
                    const std::vector<unsigned int>& indices,
                    const std::vector<glm::vec3>& startColors,
                    const std::vector<glm::vec3>& endColors,
                    const std::vector<CornerUV>& corners,
                    const MeshData& dataA, const MeshData& dataB,
                    TextureCache& textures);

    // Draw with a program built from morph.vert + shader.frag.
    void Draw(GLuint shaderProgram, const glm::mat4& viewProj, float morphT);
    bool IsInitialized() const { return vao != 0; }

private:
    // What the GPU gets per corner (the attribute locations of morph.vert)
    struct Vertex {
        glm::vec3 positionStart;   // location 0
        glm::vec3 normalStart;     // location 1
        glm::vec3 positionEnd;     // location 2
        glm::vec3 normalEnd;       // location 3
        glm::vec3 colorStart;      // location 4: tint A, or A's vertex colour if untextured
        glm::vec3 colorEnd;        // location 5
        glm::vec2 uvA;             // location 6
        glm::vec2 uvB;             // location 7
    };

    // Consecutive corners that use the same texture on each side: one draw call
    struct Range {
        int first, count;          // in corners
        int matA, matB;            // -1 = untextured on that side
    };

    GLuint vao = 0, vbo = 0;       // no index buffer: corners aren't shared
    std::vector<Range> ranges;
    std::vector<GLuint> texturesA, texturesB;   // per material of A / B (0 = untextured; owned by the TextureCache)

    void Release();
};
