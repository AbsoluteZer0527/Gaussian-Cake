#include "SplatRenderer.h"
#include "Shader.h"

#include <glm/gtc/type_ptr.hpp>
#include <algorithm>

// Per-instance layout: center(3) | cov xx,xy,xz (3) | cov yy,yz,zz (3) | rgba (4)
static const int kFloatsPerSplat = 13;

SplatRenderer::SplatRenderer()
    : isInitialized(false), program(0), vao(0), quadVbo(0), instanceVbo(0),
      instanceCapacity(0) {}

SplatRenderer::~SplatRenderer() {
    if (vao)         glDeleteVertexArrays(1, &vao);
    if (quadVbo)     glDeleteBuffers(1, &quadVbo);
    if (instanceVbo) glDeleteBuffers(1, &instanceVbo);
    if (program)     glDeleteProgram(program);
}

bool SplatRenderer::Initialize() {
    program = LoadShaders("shaders/splat.vert", "shaders/splat.frag");
    if (!program) {
        std::cerr << "SplatRenderer: failed to load splat shaders\n";
        return false;
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &quadVbo);
    glGenBuffers(1, &instanceVbo);

    glBindVertexArray(vao);

    // Static unit quad corners, drawn as a triangle strip
    const float corners[8] = { -1, -1,   1, -1,   -1, 1,   1, 1 };
    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // Per-instance attributes (divisor 1)
    glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
    const GLsizei stride = kFloatsPerSplat * sizeof(float);
    const int sizes[4]   = { 3, 3, 3, 4 };
    int offset = 0;
    for (int a = 0; a < 4; ++a) {
        GLuint loc = (GLuint)(a + 1);
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, sizes[a], GL_FLOAT, GL_FALSE, stride,
                              (void*)(offset * sizeof(float)));
        glVertexAttribDivisor(loc, 1);
        offset += sizes[a];
    }

    glBindVertexArray(0);
    isInitialized = true;
    return true;
}

void SplatRenderer::Draw(const std::vector<Splat>& splats,
                         const glm::mat4& view,
                         const glm::mat4& proj,
                         float opacityScale) {
    if (!isInitialized || splats.empty() || opacityScale <= 0.0f) return;

    const size_t n = splats.size();

    // 1. Back-to-front sort by view-space z (camera looks down -z, so the
    //    most negative z is farthest and must be drawn first).
    glm::vec4 zRow(view[0][2], view[1][2], view[2][2], view[3][2]);
    depthKeys.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const glm::vec3& m = splats[i].mean;
        float z = zRow.x * m.x + zRow.y * m.y + zRow.z * m.z + zRow.w;
        depthKeys[i] = { z, (uint32_t)i };
    }
    std::sort(depthKeys.begin(), depthKeys.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 2. Build the 3D covariance Sigma = (R S)(R S)^T per splat, in sorted order.
    instanceData.resize(n * kFloatsPerSplat);
    for (size_t k = 0; k < n; ++k) {
        const Splat& s = splats[depthKeys[k].second];
        glm::mat3 M = glm::mat3_cast(s.rot) * glm::mat3(
            s.scale.x, 0, 0,
            0, s.scale.y, 0,
            0, 0, s.scale.z);
        glm::mat3 cov = M * glm::transpose(M);

        float* d = &instanceData[k * kFloatsPerSplat];
        d[0] = s.mean.x;   d[1] = s.mean.y;   d[2] = s.mean.z;
        d[3] = cov[0][0];  d[4] = cov[1][0];  d[5] = cov[2][0];
        d[6] = cov[1][1];  d[7] = cov[2][1];  d[8] = cov[2][2];
        d[9] = s.color.r;  d[10] = s.color.g; d[11] = s.color.b;
        d[12] = s.opacity * opacityScale;
    }

    // 3. Upload (grow the buffer only when needed)
    glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
    if (n > instanceCapacity) {
        instanceCapacity = n;
        glBufferData(GL_ARRAY_BUFFER, instanceData.size() * sizeof(float),
                     instanceData.data(), GL_STREAM_DRAW);
    } else {
        glBufferSubData(GL_ARRAY_BUFFER, 0, instanceData.size() * sizeof(float),
                        instanceData.data());
    }

    // 4. Draw
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);

    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(program, "proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniform2f(glGetUniformLocation(program, "viewport"), (float)vp[2], (float)vp[3]);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)n);
    glBindVertexArray(0);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(0);
}
