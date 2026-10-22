#include "GpuSDFMorph.h"
#include "Shader.h"

#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <iostream>

// Matches the Counters block in sdf_common.glsl. The first five fields are the
// layout glDrawElementsIndirect reads.
struct SdfCounters {
    GLuint drawCount;       // indices to draw
    GLuint instanceCount;
    GLuint firstIndex;
    GLint  baseVertex;
    GLuint baseInstance;
    GLuint vertCount;       // vertices the cell pass asked for
    GLuint indexCount;      // indices the quad pass asked for
    GLuint overflow;        // 1 if either ran out of room
};

// One surface vertex as the GPU stores it (std430: two vec4s).
static const GLsizei kVertexStride = 8 * sizeof(float);
static const int kGroup = 4;   // compute workgroups are 4x4x4 threads

static GLuint Groups(int count) { return (GLuint)((count + kGroup - 1) / kGroup); }

GpuSDFMorph::~GpuSDFMorph() {
    Release();
    for (GLuint p : { progField, progCells, progQuads, progFinalize })
        if (p) glDeleteProgram(p);
}

bool GpuSDFMorph::Init() {
    if (progField) return true;
    const char* common = "shaders/sdf_common.glsl";
    progField    = LoadComputeShader("shaders/sdf_field.comp", common);
    progCells    = LoadComputeShader("shaders/sdf_cells.comp", common);
    progQuads    = LoadComputeShader("shaders/sdf_quads.comp", common);
    progFinalize = LoadComputeShader("shaders/sdf_finalize.comp", common);
    if (!progField || !progCells || !progQuads || !progFinalize) {
        std::cerr << "[GpuSDFMorph] compute shaders failed to build\n";
        return false;
    }
    return true;
}

void GpuSDFMorph::Release() {
    for (GLuint* t : { &texA, &texB })
        if (*t) { glDeleteTextures(1, t); *t = 0; }
    for (GLuint* buf : { &fieldBuf, &cellBuf, &vertBuf, &indexBuf, &counterBuf })
        if (*buf) { glDeleteBuffers(1, buf); *buf = 0; }
    for (int k = 0; k < kReadbacks; ++k) {
        if (readFence[k]) { glDeleteSync(readFence[k]); readFence[k] = 0; }
        if (readBuf[k])   { glDeleteBuffers(1, &readBuf[k]); readBuf[k] = 0; }
    }
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    ready = false;
}

static GLuint MakeGridTexture(const SDFGrid& g) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_3D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    // x fastest, then y, then z: the same order a 3D texture stores texels
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, g.n, g.n, g.n, 0, GL_RED, GL_FLOAT, g.v.data());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    for (GLenum w : { GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R })
        glTexParameteri(GL_TEXTURE_3D, w, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_3D, 0);
    return tex;
}

static GLuint MakeBuffer(GLenum target, size_t bytes, GLenum usage, const void* data = nullptr) {
    GLuint buf;
    glGenBuffers(1, &buf);
    glBindBuffer(target, buf);
    glBufferData(target, bytes, data, usage);
    glBindBuffer(target, 0);
    return buf;
}

void GpuSDFMorph::Upload(const SDFMorph& morph, int vertCapHint, int indexCapHint) {
    Release();
    if (!morph.IsBuilt() || !Init()) return;

    const SDFGrid& ga = morph.GridA();
    const SDFGrid& gb = morph.GridB();
    n  = ga.n;
    lo = ga.lo;
    h  = ga.h;
    A  = morph.MapMatrix();
    b  = morph.MapOffset();

    texA = MakeGridTexture(ga);
    texB = MakeGridTexture(gb);

    const size_t points = (size_t)n * n * n;
    const size_t cells  = (size_t)(n - 1) * (n - 1) * (n - 1);
    fieldBuf   = MakeBuffer(GL_SHADER_STORAGE_BUFFER, points * sizeof(float), GL_DYNAMIC_COPY);
    cellBuf    = MakeBuffer(GL_SHADER_STORAGE_BUFFER, cells * sizeof(GLint), GL_DYNAMIC_COPY);
    counterBuf = MakeBuffer(GL_SHADER_STORAGE_BUFFER, sizeof(SdfCounters), GL_DYNAMIC_COPY);
    for (int k = 0; k < kReadbacks; ++k)
        readBuf[k] = MakeBuffer(GL_COPY_WRITE_BUFFER, sizeof(SdfCounters), GL_STREAM_READ);

    glGenVertexArrays(1, &vao);
    Allocate(std::max(vertCapHint, 1024), std::max(indexCapHint, 6 * 1024));
    ready = true;
}

// (Re)creates the vertex and index buffers and points the VAO at them: the
// buffers the compute passes write are the same ones the draw reads.
void GpuSDFMorph::Allocate(int newVertCap, int newIndexCap) {
    if (vertBuf)  glDeleteBuffers(1, &vertBuf);
    if (indexBuf) glDeleteBuffers(1, &indexBuf);
    ++allocGen;   // readbacks still in flight describe the old buffers
    vertCap  = newVertCap;
    indexCap = newIndexCap - newIndexCap % 6;
    vertBuf  = MakeBuffer(GL_SHADER_STORAGE_BUFFER, (size_t)vertCap * kVertexStride, GL_DYNAMIC_COPY);
    indexBuf = MakeBuffer(GL_SHADER_STORAGE_BUFFER, (size_t)indexCap * sizeof(GLuint), GL_DYNAMIC_COPY);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vertBuf);
    glEnableVertexAttribArray(0);   // position (xyz of the first vec4)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, kVertexStride, (void*)0);
    glEnableVertexAttribArray(1);   // normal (xyz of the second vec4)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, kVertexStride, (void*)(4 * sizeof(float)));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void SetCommonUniforms(GLuint p, int n, const glm::vec3& lo, float h, float t,
                              const glm::mat3& LtInv, const glm::vec3& shift,
                              const glm::mat3& A, const glm::vec3& b,
                              int vertCap, int indexCap) {
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "gridA"), 0);
    glUniform1i(glGetUniformLocation(p, "gridB"), 1);
    glUniform1i(glGetUniformLocation(p, "n"), n);
    glUniform3fv(glGetUniformLocation(p, "lo"), 1, glm::value_ptr(lo));
    glUniform1f(glGetUniformLocation(p, "h"), h);
    glUniform1f(glGetUniformLocation(p, "t"), t);
    glUniformMatrix3fv(glGetUniformLocation(p, "LtInv"), 1, GL_FALSE, glm::value_ptr(LtInv));
    glUniform3fv(glGetUniformLocation(p, "shift"), 1, glm::value_ptr(shift));
    glUniformMatrix3fv(glGetUniformLocation(p, "Amat"), 1, GL_FALSE, glm::value_ptr(A));
    glUniform3fv(glGetUniformLocation(p, "bvec"), 1, glm::value_ptr(b));
    glUniform1ui(glGetUniformLocation(p, "vertCap"), (GLuint)vertCap);
    glUniform1ui(glGetUniformLocation(p, "indexCap"), (GLuint)indexCap);
}

void GpuSDFMorph::Extract(float t) {
    if (!ready) return;
    t = glm::clamp(t, 0.0f, 1.0f);

    // M_t(x) = L_t x + t b,  L_t = (1 - t) I + t A  (same as SDFMorph::Extract)
    glm::mat3 Lt    = (1.0f - t) * glm::mat3(1.0f) + t * A;
    glm::mat3 LtInv = glm::inverse(Lt);
    glm::vec3 shift = t * b;

    SdfCounters zero = { 0, 1, 0, 0, 0, 0, 0, 0 };
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterBuf);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zero), &zero);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fieldBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cellBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vertBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, indexBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, counterBuf);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, texA);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, texB);
    glActiveTexture(GL_TEXTURE0);

    auto run = [&](GLuint prog, int count) {
        SetCommonUniforms(prog, n, lo, h, t, LtInv, shift, A, b, vertCap, indexCap);
        glDispatchCompute(Groups(count), Groups(count), Groups(count));
        // a memory barrier: the next pass reads what this one wrote
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    run(progField, n);       // one thread per grid point
    run(progCells, n - 1);   // one thread per cell
    run(progQuads, n);       // one thread per grid point (its 3 edges)

    SetCommonUniforms(progFinalize, n, lo, h, t, LtInv, shift, A, b, vertCap, indexCap);
    glDispatchCompute(1, 1, 1);

    // the draw reads vertices, indices and its count from these buffers
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT |
                    GL_COMMAND_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    // copy the counters aside so PollStats can read them a few frames later
    int k = readNext;
    if (!readFence[k]) {
        glBindBuffer(GL_COPY_READ_BUFFER, counterBuf);
        glBindBuffer(GL_COPY_WRITE_BUFFER, readBuf[k]);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(SdfCounters));
        readFence[k] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        readGen[k]   = allocGen;
        readNext = (readNext + 1) % kReadbacks;
    }

    glBindTexture(GL_TEXTURE_3D, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);
}

bool GpuSDFMorph::PollStats() {
    if (!ready) return false;
    bool grew = false;
    for (int k = 0; k < kReadbacks; ++k) {
        if (!readFence[k]) continue;
        // timeout 0: only ask whether the GPU is done, never wait
        GLenum s = glClientWaitSync(readFence[k], 0, 0);
        if (s != GL_ALREADY_SIGNALED && s != GL_CONDITION_SATISFIED) continue;
        glDeleteSync(readFence[k]);
        readFence[k] = 0;

        SdfCounters c;
        glBindBuffer(GL_COPY_WRITE_BUFFER, readBuf[k]);
        glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(c), &c);
        lastVerts = (int)std::min(c.vertCount, (GLuint)vertCap);
        lastTris  = (int)c.drawCount / 3;

        // an overflow from before the last grow was already handled
        if (c.overflow && !grew && readGen[k] == allocGen) {
            int nv = std::max(vertCap * 2, (int)(c.vertCount + c.vertCount / 2));
            int ni = std::max(indexCap * 2, (int)(c.indexCount + c.indexCount / 2));
            std::cout << "[GpuSDFMorph] buffers too small, growing to " << nv << " verts, "
                      << ni / 3 << " tris\n";
            Allocate(nv, ni);
            grew = true;
        }
    }
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    return grew;
}

void GpuSDFMorph::ReadCountsNow(int& verts, int& tris, bool& overflowed) {
    verts = tris = 0;
    overflowed = false;
    if (!ready) return;
    SdfCounters c;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterBuf);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(c), &c);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    verts = (int)c.vertCount;
    tris  = (int)c.drawCount / 3;
    overflowed = c.overflow != 0;
}

void GpuSDFMorph::Draw(GLuint shaderProgram, const glm::mat4& viewProj, const glm::vec3& diffuseColor) {
    if (!ready) return;
    glUseProgram(shaderProgram);
    glm::mat4 model(1.0f);
    GLint locVP    = glGetUniformLocation(shaderProgram, "viewProj");
    GLint locM     = glGetUniformLocation(shaderProgram, "model");
    GLint locDiff  = glGetUniformLocation(shaderProgram, "DiffuseColor");
    GLint locAlpha = glGetUniformLocation(shaderProgram, "alpha");
    if (locVP    >= 0) glUniformMatrix4fv(locVP, 1, GL_FALSE, glm::value_ptr(viewProj));
    if (locM     >= 0) glUniformMatrix4fv(locM, 1, GL_FALSE, glm::value_ptr(model));
    if (locDiff  >= 0) glUniform3fv(locDiff, 1, glm::value_ptr(diffuseColor));
    if (locAlpha >= 0) glUniform1f(locAlpha, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(vao);
    // the index count comes from the GPU-written command, not from the CPU
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, counterBuf);
    glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, (void*)0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}
