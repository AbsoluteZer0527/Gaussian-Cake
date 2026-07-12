#pragma once
#include "core.h"
#include <vector>

// Renders a deforming triangle mesh via a modern VAO/VBO/EBO pipeline.
// Topology (indices) is uploaded once. Positions are updated per frame
// as the mesh morphs; normals are recomputed each update for smooth Phong shading.
class MeshRenderer {
public:
    MeshRenderer();
    ~MeshRenderer();

    // Upload initial vertices and triangle connectivity.
    // indices.size() must be a multiple of 3.
    void Initialize(const std::vector<glm::vec3>& positions,
                    const std::vector<unsigned int>& indices);

    // Replace vertex positions (must match count from Initialize).
    // Recomputes area-weighted vertex normals and uploads both buffers.
    void UpdatePositions(const std::vector<glm::vec3>& positions);

    // Render as GL_TRIANGLES using the provided shader program.
    void Draw(GLuint shaderProgram, const glm::mat4& viewProj);

    bool isInitialized;

private:
    GLuint vao;
    GLuint vboPos;
    GLuint vboNorm;
    GLuint ebo;

    int vertexCount;
    int indexCount;

    // Kept CPU-side so normals can be recomputed on each UpdatePositions.
    std::vector<unsigned int> triIndices;

    void ComputeNormals(const std::vector<glm::vec3>& positions,
                        std::vector<glm::vec3>& outNormals) const;
};
