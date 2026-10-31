#pragma once
#include "core.h"
#include <vector>

// Renders a deforming triangle mesh via a modern VAO/VBO/EBO pipeline.
// Topology (indices) is uploaded once. Positions are updated per frame
// as the mesh morphs; normals are recomputed each update for smooth Phong shading.
//
// Morph-target mode (InitializeMorph): both end shapes are uploaded once and
// the vertex shader blends them with the morphT uniform, so a frame of the
// morph costs the CPU nothing but one float (what game engines call blend
// shapes / morph targets). Draw it with a program built from morph.vert.
class MeshRenderer {
public:
    MeshRenderer();
    ~MeshRenderer();

    // Upload initial vertices and triangle connectivity.
    // indices.size() must be a multiple of 3.
    // colors: optional per-vertex display colours (empty = diffuseColor everywhere).
    void Initialize(const std::vector<glm::vec3>& positions,
                    const std::vector<unsigned int>& indices,
                    const std::vector<glm::vec3>& colors = {});

    // Upload two shapes with the same connectivity: start at morphT = 0, end at
    // morphT = 1. Normals of both ends are precomputed here.
    // Colours, if given, blend from startColors to endColors the same way.
    void InitializeMorph(const std::vector<glm::vec3>& startPositions,
                         const std::vector<glm::vec3>& endPositions,
                         const std::vector<unsigned int>& indices,
                         const std::vector<glm::vec3>& startColors = {},
                         const std::vector<glm::vec3>& endColors = {});

    // Replace vertex positions (must match count from Initialize).
    // Recomputes area-weighted vertex normals and uploads both buffers.
    // colors: new per-vertex colours too, if the mesh has them.
    void UpdatePositions(const std::vector<glm::vec3>& positions,
                         const std::vector<glm::vec3>& colors = {});

    // Render as GL_TRIANGLES using the provided shader program.
    void Draw(GLuint shaderProgram, const glm::mat4& viewProj);

    // Diffuse color and opacity passed to the fragment shader. Public — set directly.
    glm::vec3 diffuseColor;
    float     alpha;          // 0.0 = fully transparent, 1.0 = fully opaque
    float     morphT;         // blend between the two shapes in morph-target mode

    bool isInitialized;
    bool IsMorphTarget() const { return morphTarget; }
    bool HasColors() const { return hasColors; }
    int  VertexCount() const { return vertexCount; }
    int  TriangleCount() const { return indexCount / 3; }

private:
    GLuint vao;
    GLuint vboPos;
    GLuint vboNorm;
    GLuint vboPosEnd;     // morph-target mode only
    GLuint vboNormEnd;
    GLuint vboColor;      // per-vertex colours (location 4), end colours (location 5)
    GLuint vboColorEnd;
    GLuint ebo;

    int vertexCount;
    int indexCount;
    bool morphTarget;
    bool hasColors;

    // Kept CPU-side so normals can be recomputed on each UpdatePositions.
    std::vector<unsigned int> triIndices;

    void Upload(GLuint vbo, GLuint location, const std::vector<glm::vec3>& data);
    void ComputeNormals(const std::vector<glm::vec3>& positions,
                        std::vector<glm::vec3>& outNormals) const;
};
