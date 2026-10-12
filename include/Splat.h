#pragma once
#include "core.h"
#include <glm/gtc/quaternion.hpp>

// One 3D Gaussian splat. Covariance is stored as rotation + per-axis std-dev
// (Sigma = R * S^2 * R^T) so it can be interpolated cleanly even for very
// flat (disk-shaped) splats, where the raw covariance is nearly singular.
// Convention: local z axis is the splat normal (the thin axis).
struct Splat {
    glm::vec3 mean    = glm::vec3(0.0f);
    glm::quat rot     = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale   = glm::vec3(0.01f);
    glm::vec3 color   = glm::vec3(1.0f);
    float     opacity = 1.0f;
};
