#pragma once
#include "core.h"
#include "SDFMorph.h"

// GPU version of SDFMorph::Extract + drawing (needs OpenGL 4.3).
//
// The two distance grids become 3D textures, so the hardware does the
// trilinear lookups. Each frame runs four compute passes, each one thread per
// loop iteration of the CPU version:
//   1. field:    phi_t at every grid point
//   2. cells:    one surface-nets vertex per cell the surface crosses, with a
//                normal from the field gradient
//   3. quads:    one quad per grid edge with a sign change
//   4. finalize: write the triangle count into an indirect draw command
// Vertices and indices get their slots from atomic counters, and the draw
// reads its count from GPU memory, so the CPU never waits for the result.
class GpuSDFMorph {
public:
    ~GpuSDFMorph();

    // Compiles the compute programs. Returns false if they fail to build.
    bool Init();

    // Upload both grids and the OT map. vertCap / indexCap: initial buffer
    // sizes (they grow on their own if a frame needs more).
    void Upload(const SDFMorph& morph, int vertCap, int indexCap);

    // Build the t-surface on the GPU.
    void Extract(float t);

    // Draw the last extracted surface with the regular mesh shader.
    void Draw(GLuint shaderProgram, const glm::mat4& viewProj, const glm::vec3& diffuseColor);

    // Reads counts from extractions a few frames old (no stall). Returns true
    // if a frame overflowed and the buffers were grown, so the caller should
    // extract again.
    bool PollStats();

    // Counts of the latest extraction, read right away (waits for the GPU;
    // for checking against the CPU version, not for every frame).
    void ReadCountsNow(int& verts, int& tris, bool& overflowed);

    bool IsReady() const { return ready; }
    int  LastVertexCount() const { return lastVerts; }
    int  LastTriangleCount() const { return lastTris; }
    int  Resolution() const { return n; }

private:
    void Allocate(int vertCap, int indexCap);
    void Release();

    GLuint progField = 0, progCells = 0, progQuads = 0, progFinalize = 0;
    GLuint texA = 0, texB = 0;
    GLuint fieldBuf = 0, cellBuf = 0, vertBuf = 0, indexBuf = 0, counterBuf = 0;
    GLuint vao = 0;

    static const int kReadbacks = 3;
    GLuint readBuf[kReadbacks] = {};
    GLsync readFence[kReadbacks] = {};
    int    readGen[kReadbacks] = {};   // allocGen when the readback was issued
    int    readNext = 0;
    int    allocGen = 0;               // bumped whenever the buffers are reallocated

    int n = 0;
    glm::vec3 lo = glm::vec3(0.0f);
    float h = 1.0f;
    glm::mat3 A = glm::mat3(1.0f);
    glm::vec3 b = glm::vec3(0.0f);
    int vertCap = 0, indexCap = 0;

    bool ready = false;
    int lastVerts = 0, lastTris = 0;
};
