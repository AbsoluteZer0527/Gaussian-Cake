#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

// Squared distance between two points (no square root: enough to compare distances)
inline float Dist2(const glm::vec3& a, const glm::vec3& b) {
    glm::vec3 d = a - b;
    return glm::dot(d, d);
}
