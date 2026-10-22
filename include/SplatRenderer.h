#pragma once
#include "core.h"
#include "Splat.h"
#include "GpuTimer.h"
#include <vector>
#include <cstdint>

// Renders 3D Gaussian splats with EWA splatting (Zwicker et al. 2001, as used
// by 3DGS): each splat is projected to a 2D screen-space Gaussian, drawn as an
// instanced quad, sorted back-to-front and alpha-blended.
//
// Two paths:
//   Draw():      splats come from the CPU each frame and are sorted there.
//   DrawMorph(): the morph keyframes were uploaded once (UploadKeyframes) and
//                everything per frame runs on the GPU (needs OpenGL 4.3):
//                interpolate + project (splat_prep.comp), radix sort by depth
//                (radix_*.comp), draw (splat_gpu.vert).
class SplatRenderer {
public:
    SplatRenderer();
    ~SplatRenderer();

    // Load splat shaders and allocate GL objects. Call once a GL context exists.
    bool Initialize();

    // Sort + upload + draw. opacityScale multiplies every splat's opacity
    // (used for the A/B crossfade during a morph).
    void Draw(const std::vector<Splat>& splats,
              const glm::mat4& view,
              const glm::mat4& proj,
              float opacityScale = 1.0f);

    // ---- GPU morph path ----
    // Compiles the compute programs; false if unavailable.
    bool InitializeGpu();

    // start[i] travels to end[i]; smoothDisp may be empty (straight paths).
    void UploadKeyframes(const std::vector<Splat>& start, const std::vector<Splat>& end,
                         const std::vector<glm::vec3>& smoothDisp);

    // e = eased time, g = cohesion weight, grow = mid-flight size factor
    // (the same values the CPU loop in Window::idleCallback uses).
    void DrawMorph(float e, float g, float grow, const glm::mat4& view, const glm::mat4& proj);

    bool HasKeyframes() const { return gpuCount > 0; }

    // Splats sorted on the GPU last frame, back to front (debug readback; stalls).
    bool CheckGpuSortOrder(std::vector<uint32_t>* keysOut = nullptr);

    // GPU time (ms, smoothed) of interpolate + sort, and of the draw
    double PrepSortMs() const { return prepTimer.Ms(); }
    double DrawMs() const { return drawTimer.Ms(); }

    bool isInitialized;

private:
    GLuint program;
    GLuint vao;
    GLuint quadVbo;
    GLuint instanceVbo;
    size_t instanceCapacity;   // in splats

    // CPU staging, reused across frames to avoid reallocations
    std::vector<std::pair<float, uint32_t>> depthKeys;
    std::vector<float> instanceData;

    // GPU morph path
    GLuint gpuDrawProgram = 0, progPrep = 0, progHist = 0, progScan = 0, progScatter = 0;
    GLuint gpuVao = 0;          // only the quad corners: per-splat data comes from buffers
    GLuint keyframeBuf = 0, prepBuf = 0, histBuf = 0;
    GLuint sortKeys[2] = {}, sortVals[2] = {};
    int    gpuCount = 0;
    bool   gpuHasSmooth = false;
    void   ReleaseGpuBuffers();
    void   SortByDepth();

    GpuTimer prepTimer, drawTimer;

    void   SetBlendState(bool begin);
};
