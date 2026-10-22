#include "SplatRenderer.h"
#include "Shader.h"

#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <iostream>

// Per-instance layout: center(3) | cov xx,xy,xz (3) | cov yy,yz,zz (3) | rgba (4)
static const int kFloatsPerSplat = 13;

SplatRenderer::SplatRenderer()
    : isInitialized(false), program(0), vao(0), quadVbo(0), instanceVbo(0),
      instanceCapacity(0) {}

SplatRenderer::~SplatRenderer() {
    ReleaseGpuBuffers();
    if (gpuVao)      glDeleteVertexArrays(1, &gpuVao);
    for (GLuint p : { gpuDrawProgram, progPrep, progHist, progScan, progScatter })
        if (p) glDeleteProgram(p);
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

    SetBlendState(true);
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)n);
    glBindVertexArray(0);
    SetBlendState(false);
    glUseProgram(0);
}

// Splats are transparent, so they follow the usual rule for transparent
// particles: depth test on (anything opaque drawn earlier hides them), depth
// write off (they don't hide each other; the back-to-front sort orders them).
void SplatRenderer::SetBlendState(bool begin) {
    if (begin) {
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDisable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    } else {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
}

// ---- GPU morph path ----

// Matches SplatKeyframes in splat_prep.comp (std430: 9 vec4s).
struct GpuSplatKeyframes {
    glm::vec4 meanA, meanB;
    glm::vec4 rotA, rotB;             // (x, y, z, w)
    glm::vec4 logScaleA, logScaleB;   // w = opacity
    glm::vec4 colorA, colorB;
    glm::vec4 smoothDisp;
};
static const size_t kPreparedBytes = 12 * sizeof(float);   // Prepared in splat_prep.comp
static const GLuint kSortBlock = 1024;                     // kBlock in radix_common.glsl

bool SplatRenderer::InitializeGpu() {
    if (gpuDrawProgram) return true;
    progPrep    = LoadComputeShader("shaders/splat_prep.comp");
    progHist    = LoadComputeShader("shaders/radix_hist.comp", "shaders/radix_common.glsl");
    progScan    = LoadComputeShader("shaders/radix_scan.comp", "shaders/radix_common.glsl");
    progScatter = LoadComputeShader("shaders/radix_scatter.comp", "shaders/radix_common.glsl");
    gpuDrawProgram = LoadShaders("shaders/splat_gpu.vert", "shaders/splat.frag");
    if (!progPrep || !progHist || !progScan || !progScatter || !gpuDrawProgram) {
        std::cerr << "SplatRenderer: GPU splat shaders failed to build\n";
        for (GLuint* p : { &progPrep, &progHist, &progScan, &progScatter, &gpuDrawProgram })
            if (*p) { glDeleteProgram(*p); *p = 0; }
        return false;
    }

    glGenVertexArrays(1, &gpuVao);
    glBindVertexArray(gpuVao);
    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return true;
}

void SplatRenderer::ReleaseGpuBuffers() {
    for (GLuint* b : { &keyframeBuf, &prepBuf, &histBuf, &sortKeys[0], &sortKeys[1],
                       &sortVals[0], &sortVals[1] })
        if (*b) { glDeleteBuffers(1, b); *b = 0; }
    gpuCount = 0;
}

static GLuint MakeStorage(size_t bytes, const void* data = nullptr) {
    GLuint buf;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, std::max<size_t>(bytes, 16), data, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return buf;
}

void SplatRenderer::UploadKeyframes(const std::vector<Splat>& start, const std::vector<Splat>& end,
                                    const std::vector<glm::vec3>& smoothDisp) {
    ReleaseGpuBuffers();
    const size_t n = std::min(start.size(), end.size());
    if (n == 0 || !InitializeGpu()) return;
    gpuHasSmooth = smoothDisp.size() == n;

    // log(scale) is precomputed: the shader lerps it and takes exp
    auto quat = [](const glm::quat& q) { return glm::vec4(q.x, q.y, q.z, q.w); };
    std::vector<GpuSplatKeyframes> k(n);
    for (size_t i = 0; i < n; ++i) {
        const Splat& a = start[i];
        const Splat& b = end[i];
        k[i].meanA      = glm::vec4(a.mean, 1.0f);
        k[i].meanB      = glm::vec4(b.mean, 1.0f);
        k[i].rotA       = quat(a.rot);
        k[i].rotB       = quat(b.rot);
        k[i].logScaleA  = glm::vec4(glm::log(a.scale), a.opacity);
        k[i].logScaleB  = glm::vec4(glm::log(b.scale), b.opacity);
        k[i].colorA     = glm::vec4(a.color, 1.0f);
        k[i].colorB     = glm::vec4(b.color, 1.0f);
        k[i].smoothDisp = glm::vec4(gpuHasSmooth ? smoothDisp[i] : glm::vec3(0.0f), 0.0f);
    }

    const GLuint blocks = (GLuint)((n + kSortBlock - 1) / kSortBlock);
    keyframeBuf = MakeStorage(n * sizeof(GpuSplatKeyframes), k.data());
    prepBuf     = MakeStorage(n * kPreparedBytes);
    histBuf     = MakeStorage((size_t)256 * blocks * sizeof(GLuint));
    for (int p = 0; p < 2; ++p) {
        sortKeys[p] = MakeStorage(n * sizeof(GLuint));
        sortVals[p] = MakeStorage(n * sizeof(GLuint));
    }
    gpuCount = (int)n;
}

// LSD radix sort of (key, splat index) pairs, 8 bits per pass. Each pass:
// histogram -> prefix sum -> stable scatter, ping-ponging between the two
// buffer pairs. After 4 passes (an even number) the result is back in pair 0.
void SplatRenderer::SortByDepth() {
    const GLuint n = (GLuint)gpuCount;
    const GLuint blocks = (n + kSortBlock - 1) / kSortBlock;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, histBuf);

    for (GLuint pass = 0; pass < 4; ++pass) {
        int in = pass % 2, out = 1 - in;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sortKeys[in]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sortVals[in]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, sortKeys[out]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, sortVals[out]);

        for (GLuint prog : { progHist, progScan, progScatter }) {
            glUseProgram(prog);
            glUniform1ui(glGetUniformLocation(prog, "count"), n);
            glUniform1ui(glGetUniformLocation(prog, "shift"), pass * 8);
            glUniform1ui(glGetUniformLocation(prog, "numBlocks"), blocks);
            glDispatchCompute(prog == progScan ? 1 : blocks, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
}

void SplatRenderer::DrawMorph(float e, float g, float grow,
                              const glm::mat4& view, const glm::mat4& proj) {
    if (!HasKeyframes()) return;
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    glm::vec2 viewport((float)vp[2], (float)vp[3]);

    prepTimer.Begin();
    // 1. interpolate + project every splat, write its depth key
    glUseProgram(progPrep);
    glUniform1ui(glGetUniformLocation(progPrep, "count"), (GLuint)gpuCount);
    glUniform1f(glGetUniformLocation(progPrep, "e"), e);
    glUniform1f(glGetUniformLocation(progPrep, "g"), g);
    glUniform1f(glGetUniformLocation(progPrep, "grow"), grow);
    glUniform1i(glGetUniformLocation(progPrep, "hasSmooth"), gpuHasSmooth ? 1 : 0);
    glUniformMatrix4fv(glGetUniformLocation(progPrep, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(progPrep, "proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniform2fv(glGetUniformLocation(progPrep, "viewport"), 1, glm::value_ptr(viewport));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, keyframeBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, prepBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, sortKeys[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, sortVals[0]);
    glDispatchCompute((GLuint)((gpuCount + 255) / 256), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 2. back-to-front order
    SortByDepth();
    prepTimer.End();

    // 3. one instanced quad per splat, in sorted order
    drawTimer.Begin();
    glUseProgram(gpuDrawProgram);
    glUniform2fv(glGetUniformLocation(gpuDrawProgram, "viewport"), 1, glm::value_ptr(viewport));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, prepBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sortVals[0]);
    SetBlendState(true);
    glBindVertexArray(gpuVao);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)gpuCount);
    glBindVertexArray(0);
    SetBlendState(false);
    drawTimer.End();

    for (GLuint b = 0; b < 5; ++b) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, b, 0);
    glUseProgram(0);
}

bool SplatRenderer::CheckGpuSortOrder(std::vector<uint32_t>* keysOut) {
    if (!HasKeyframes()) return false;
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    std::vector<uint32_t> keys(gpuCount), vals(gpuCount);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortKeys[0]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, keys.size() * sizeof(uint32_t), keys.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortVals[0]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vals.size() * sizeof(uint32_t), vals.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    bool ok = true;
    for (size_t i = 1; i < keys.size(); ++i)
        if (keys[i - 1] > keys[i]) { ok = false; break; }
    // every splat appears exactly once
    std::vector<char> seen(gpuCount, 0);
    for (uint32_t v : vals) {
        if (v >= (uint32_t)gpuCount || seen[v]) { ok = false; break; }
        seen[v] = 1;
    }
    if (keysOut) keysOut->swap(keys);
    return ok;
}
