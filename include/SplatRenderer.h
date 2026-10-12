#pragma once
#include "core.h"
#include "Splat.h"
#include <vector>
#include <cstdint>

// Renders 3D Gaussian splats with EWA splatting (Zwicker et al. 2001, as used
// by 3DGS): each splat is projected to a 2D screen-space Gaussian, drawn as an
// instanced quad, sorted back-to-front on the CPU and alpha-blended.
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
};
