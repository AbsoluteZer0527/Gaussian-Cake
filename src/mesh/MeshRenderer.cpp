#include "MeshRenderer.h"

#include <glm/gtc/type_ptr.hpp>
#include <iostream>

MeshRenderer::MeshRenderer()
    : diffuseColor(0.72f, 0.72f, 0.74f),  // default neutral gray
      alpha(1.0f), morphT(0.0f),
      isInitialized(false), vao(0), vboPos(0), vboNorm(0), vboPosEnd(0), vboNormEnd(0),
      vboColor(0), vboColorEnd(0), ebo(0),
      vertexCount(0), indexCount(0), morphTarget(false), hasColors(false) {}

MeshRenderer::~MeshRenderer() {
    if (vao)        glDeleteVertexArrays(1, &vao);
    if (vboPos)     glDeleteBuffers(1, &vboPos);
    if (vboNorm)    glDeleteBuffers(1, &vboNorm);
    if (vboPosEnd)  glDeleteBuffers(1, &vboPosEnd);
    if (vboNormEnd) glDeleteBuffers(1, &vboNormEnd);
    if (vboColor)   glDeleteBuffers(1, &vboColor);
    if (vboColorEnd) glDeleteBuffers(1, &vboColorEnd);
    if (ebo)        glDeleteBuffers(1, &ebo);
}

// Upload one vec3 attribute buffer and bind it to a vertex attribute location
// (the VAO must be bound).
void MeshRenderer::Upload(GLuint vbo, GLuint location, const std::vector<glm::vec3>& data) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(glm::vec3), data.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(location);
    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
}

void MeshRenderer::Initialize(const std::vector<glm::vec3>& positions,
                              const std::vector<unsigned int>& indices,
                              const std::vector<glm::vec3>& colors) {
    if (positions.empty() || indices.empty() || indices.size() % 3 != 0) {
        std::cerr << "MeshRenderer::Initialize: invalid input\n";
        return;
    }

    vertexCount = (int)positions.size();
    indexCount  = (int)indices.size();
    triIndices  = indices;
    morphTarget = false;

    // Lazy allocation on first use — safe because GL context now exists.
    if (!vao) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vboPos);
        glGenBuffers(1, &vboNorm);
        glGenBuffers(1, &ebo);
    }

    glBindVertexArray(vao);

    // Position buffer at location = 0, normals at location = 1
    std::vector<glm::vec3> normals(vertexCount);
    ComputeNormals(positions, normals);
    Upload(vboPos, 0, positions);
    Upload(vboNorm, 1, normals);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(5);

    // Colours at location 4, if there's one per vertex
    hasColors = (colors.size() == positions.size());
    if (hasColors) {
        if (!vboColor) glGenBuffers(1, &vboColor);
        Upload(vboColor, 4, colors);
    } else {
        glDisableVertexAttribArray(4);
    }

    // Triangle indices (topology stays fixed for the entire morph)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indexCount * sizeof(unsigned int),
                 indices.data(),
                 GL_STATIC_DRAW);

    glBindVertexArray(0);
    isInitialized = true;
}

void MeshRenderer::InitializeMorph(const std::vector<glm::vec3>& startPositions,
                                   const std::vector<glm::vec3>& endPositions,
                                   const std::vector<unsigned int>& indices,
                                   const std::vector<glm::vec3>& startColors,
                                   const std::vector<glm::vec3>& endColors) {
    if (startPositions.size() != endPositions.size()) {
        std::cerr << "MeshRenderer::InitializeMorph: start/end vertex counts differ\n";
        return;
    }
    bool colored = startColors.size() == startPositions.size() &&
                   endColors.size() == endPositions.size();
    Initialize(startPositions, indices, colored ? startColors : std::vector<glm::vec3>());
    if (!isInitialized) return;

    if (!vboPosEnd) {
        glGenBuffers(1, &vboPosEnd);
        glGenBuffers(1, &vboNormEnd);
    }
    std::vector<glm::vec3> normals(vertexCount);
    ComputeNormals(endPositions, normals);

    // End shape at locations 2 (position) and 3 (normal)
    glBindVertexArray(vao);
    Upload(vboPosEnd, 2, endPositions);
    Upload(vboNormEnd, 3, normals);
    if (colored) {
        if (!vboColorEnd) glGenBuffers(1, &vboColorEnd);
        Upload(vboColorEnd, 5, endColors);
    }
    glBindVertexArray(0);
    morphTarget = true;
}

void MeshRenderer::UpdatePositions(const std::vector<glm::vec3>& positions,
                                   const std::vector<glm::vec3>& colors) {
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

    if (hasColors && (int)colors.size() == vertexCount) {
        glBindBuffer(GL_ARRAY_BUFFER, vboColor);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertexCount * sizeof(glm::vec3), colors.data());
    }
}

void MeshRenderer::Draw(GLuint shaderProgram, const glm::mat4& viewProj) {
    if (!isInitialized) return;

    glUseProgram(shaderProgram);

    glm::mat4 model(1.0f);

    GLint locVP    = glGetUniformLocation(shaderProgram, "viewProj");
    GLint locM     = glGetUniformLocation(shaderProgram, "model");
    GLint locDiff  = glGetUniformLocation(shaderProgram, "DiffuseColor");
    GLint locAlpha = glGetUniformLocation(shaderProgram, "alpha");
    GLint locMorph = glGetUniformLocation(shaderProgram, "morphT");
    GLint locColor = glGetUniformLocation(shaderProgram, "useVertexColor");

    if (locVP    >= 0) glUniformMatrix4fv(locVP,  1, GL_FALSE, glm::value_ptr(viewProj));
    if (locM     >= 0) glUniformMatrix4fv(locM,   1, GL_FALSE, glm::value_ptr(model));
    if (locDiff  >= 0) glUniform3fv(locDiff, 1, glm::value_ptr(diffuseColor));
    if (locAlpha >= 0) glUniform1f(locAlpha, alpha);
    if (locMorph >= 0) glUniform1f(locMorph, morphT);
    if (locColor >= 0) glUniform1i(locColor, hasColors ? 1 : 0);

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
