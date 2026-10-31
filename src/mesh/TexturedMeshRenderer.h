#pragma once
#include "core.h"
#include "MeshData.h"
#include <unordered_map>
#include <vector>

// Draws a MeshData the way its file intends: texture coordinates and a GPU
// texture per material (sampled per pixel, so the full texture detail shows),
// times the material colour and the vertex colours. Used for the real meshes
// at rest; in-between morph surfaces have no UVs and use vertex colours.
//
// Vertices are welded where position, uv, colour and material all match, so
// UV seams stay split; normals are averaged over every vertex at the same
// position, so the seams don't show in the shading. Draw with the regular
// mesh shader (shader.vert + shader.frag).
class TexturedMeshRenderer {
public:
    ~TexturedMeshRenderer();

    void Initialize(const MeshData& mesh);
    void Draw(GLuint shaderProgram, const glm::mat4& viewProj);
    bool IsInitialized() const { return vao != 0; }

private:
    struct Range { int first, count, material; };   // indices drawn with one material

    GLuint vao = 0, vbo = 0, ebo = 0;
    std::vector<Range> ranges;
    std::vector<glm::vec3> materialColors;
    std::vector<GLuint> materialTextures;           // 0 = untextured
    std::unordered_map<const Texture*, GLuint> uploaded;   // one GL texture per image

    void Release();
    GLuint Upload(const Texture& tex);
};
