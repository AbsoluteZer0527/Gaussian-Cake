#include "MeshRenderer.h"

#include <glm/gtc/type_ptr.hpp>
#include <iostream>

MeshRenderer::MeshRenderer()
    : isInitialized(false), vao(0), vboPos(0), vboNorm(0), ebo(0),
      vertexCount(0), indexCount(0) {}

MeshRenderer::~MeshRenderer() {
    if (vao)     glDeleteVertexArrays(1, &vao);
    if (vboPos)  glDeleteBuffers(1, &vboPos);
    if (vboNorm) glDeleteBuffers(1, &vboNorm);
    if (ebo)     glDeleteBuffers(1, &ebo);
}

void MeshRenderer::Initialize(const std::vector<glm::vec3>& positions,
                              const std::vector<unsigned int>& indices) {
    if (positions.empty() || indices.empty() || indices.size() % 3 != 0) {
        std::cerr << "MeshRenderer::Initialize: invalid input\n";
        return;
    }

    vertexCount = (int)positions.size();
    indexCount  = (int)indices.size();
    triIndices  = indices;

    // Lazy allocation on first use — safe because GL context now exists.
    if (!vao) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vboPos);
        glGenBuffers(1, &vboNorm);
        glGenBuffers(1, &ebo);
    }

    glBindVertexArray(vao);

    // Position buffer at location = 0
    glBindBuffer(GL_ARRAY_BUFFER, vboPos);
    glBufferData(GL_ARRAY_BUFFER,
                 vertexCount * sizeof(glm::vec3),
                 positions.data(),
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    // Normal buffer at location = 1 (initialized zero, filled by UpdatePositions)
    std::vector<glm::vec3> initialNormals(vertexCount, glm::vec3(0.0f, 1.0f, 0.0f));
    ComputeNormals(positions, initialNormals);

    glBindBuffer(GL_ARRAY_BUFFER, vboNorm);
    glBufferData(GL_ARRAY_BUFFER,
                 vertexCount * sizeof(glm::vec3),
                 initialNormals.data(),
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    // Triangle indices (topology stays fixed for the entire morph)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indexCount * sizeof(unsigned int),
                 indices.data(),
                 GL_STATIC_DRAW);

    glBindVertexArray(0);
    isInitialized = true;
}

void MeshRenderer::UpdatePositions(const std::vector<glm::vec3>& positions) {
    if (!isInitialized || (int)positions.size() != vertexCount) return;

    // Positions
    glBindBuffer(GL_ARRAY_BUFFER, vboPos);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    vertexCount * sizeof(glm::vec3),
                    positions.data());

    // Normals (recomputed from current triangle geometry)
    std::vector<glm::vec3> normals(vertexCount, glm::vec3(0.0f));
    ComputeNormals(positions, normals);

    glBindBuffer(GL_ARRAY_BUFFER, vboNorm);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    vertexCount * sizeof(glm::vec3),
                    normals.data());
}

void MeshRenderer::Draw(GLuint shaderProgram, const glm::mat4& viewProj) {
    if (!isInitialized) return;

    glUseProgram(shaderProgram);

    glm::mat4 model(1.0f);
    glm::vec3 diffuse(0.85f, 0.55f, 0.75f);  // pink-ish, matches the app aesthetic

    GLint locVP    = glGetUniformLocation(shaderProgram, "viewProj");
    GLint locM     = glGetUniformLocation(shaderProgram, "model");
    GLint locDiff  = glGetUniformLocation(shaderProgram, "DiffuseColor");

    if (locVP   >= 0) glUniformMatrix4fv(locVP,  1, GL_FALSE, glm::value_ptr(viewProj));
    if (locM    >= 0) glUniformMatrix4fv(locM,   1, GL_FALSE, glm::value_ptr(model));
    if (locDiff >= 0) glUniform3fv(locDiff, 1, glm::value_ptr(diffuse));

    // Fixed-function lighting doesn't affect shader path, but disable it to be safe.
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glUseProgram(0);
}

void MeshRenderer::ComputeNormals(const std::vector<glm::vec3>& positions,
                                  std::vector<glm::vec3>& outNormals) const {
    // Zero out accumulator
    std::fill(outNormals.begin(), outNormals.end(), glm::vec3(0.0f));

    // For each triangle: unnormalized face normal magnitude = 2*area,
    // so summing across incident triangles gives an area-weighted average.
    for (size_t t = 0; t + 2 < triIndices.size(); t += 3) {
        unsigned int i0 = triIndices[t + 0];
        unsigned int i1 = triIndices[t + 1];
        unsigned int i2 = triIndices[t + 2];

        const glm::vec3& v0 = positions[i0];
        const glm::vec3& v1 = positions[i1];
        const glm::vec3& v2 = positions[i2];

        glm::vec3 faceN = glm::cross(v1 - v0, v2 - v0);

        outNormals[i0] += faceN;
        outNormals[i1] += faceN;
        outNormals[i2] += faceN;
    }

    // Normalize per vertex (guard against isolated / degenerate vertices)
    for (auto& n : outNormals) {
        float len2 = glm::dot(n, n);
        n = (len2 > 1e-20f) ? (n * (1.0f / sqrtf(len2))) : glm::vec3(0.0f, 1.0f, 0.0f);
    }
}
